from .provider import ProviderAdapter
import os
from dotenv import load_dotenv
import json
from openai import OpenAI
from datetime import datetime, timezone
from src.schemas import APIType, AttemptMetadata, Choice, Message, Usage, Cost, CompletionTokensDetails, Attempt
from typing import Optional, Any, List, Dict
from .openai_base import OpenAIBaseAdapter
import re

load_dotenv()


class OpenAIAdapter(OpenAIBaseAdapter):
    """Adapter specific to official OpenAI API endpoints and response structures."""

    PROVIDER_API_KEYS = {
        "openai": "OPENAI_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "fireworks": "FIREWORKS_API_KEY",
        "huggingfacefireworks": "FIREWORKS_API_KEY", # HF uses Fireworks key
        # Add other provider keys here (e.g., "groq": "GROQ_API_KEY")
    }

    def init_client(self):
        """
        Initialize the OpenAI client using the OPENAI_API_KEY.
        Uses the default base URL provided by the OpenAI library.
        """
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        # Pass base_url=None to use the default OpenAI URL
        client = OpenAI(api_key=api_key, base_url=None)
        return client


    def make_prediction(self, prompt: str, task_id: Optional[str] = None, test_id: Optional[str] = None, pair_index: int = None) -> Attempt:
        """
        Make a prediction using the OpenAI model (calling the appropriate API type via base class)
        and parse the specific OpenAI response structure.
        """
        start_time = datetime.now(timezone.utc)
        
        # Use the inherited call_ai_model which handles chat vs responses API
        response = self.call_ai_model(prompt)
        
        end_time = datetime.now(timezone.utc)

        # Use pricing from model config
        input_cost_per_token = self.model_config.pricing.input / 1_000_000
        output_cost_per_token = self.model_config.pricing.output / 1_000_000
        
        # Use the specific _get_usage implementation for OpenAI
        usage = self._get_usage(response)
        
        prompt_cost = usage.prompt_tokens * input_cost_per_token
        completion_cost = usage.completion_tokens * output_cost_per_token

        # Convert input messages to choices
        input_choices = [
            Choice(
                index=0,
                message=Message(role="user", content=prompt)
            )
        ]

        # Convert OpenAI response to our schema using specific helpers
        response_choices = [
            Choice(
                index=1,
                message=Message(
                    role=self._get_role(response),
                    content=self._get_content(response)
                )
            )
        ]

        all_choices = input_choices + response_choices

        metadata = AttemptMetadata(
            model=self.model_config.model_name,
            provider=self.model_config.provider,
            start_timestamp=start_time,
            end_timestamp=end_time,
            choices=all_choices,
            kwargs=self.model_config.kwargs,
            usage=usage,
            cost=Cost(
                prompt_cost=prompt_cost,
                completion_cost=completion_cost,
                total_cost=prompt_cost + completion_cost
            ),
            task_id=task_id,
            pair_index=pair_index,
            test_id=test_id
        )

        attempt = Attempt(
            metadata=metadata,
            answer=self._get_content(response) # Get content again for the final answer field
        )

        return attempt

    def call_ai_model(self, prompt: str):
        """
        Call the appropriate OpenAI API based on the api_type
        """
        messages = [{"role": "user", "content": prompt}]
        if self.model_config.api_type == APIType.CHAT_COMPLETIONS:
            return self.chat_completion(messages)
        else:  # APIType.RESPONSES
            # account for different parameter names between chat completions and responses APIs
            self._normalize_to_responses_kwargs()
            return self.responses(messages)
    
    def chat_completion(self, messages: list) -> str:
        """
        Make a call to the OpenAI Chat Completions API
        """
        return self.client.chat.completions.create(
            model=self.model_config.model_name,
            messages=messages,
            **self.model_config.kwargs
        )
    
    def responses(self, messages: list) -> str:
        """
        Make a call to the OpenAI Responses API
        """
        return self.client.responses.create(
            model=self.model_config.model_name,
            input=messages,
            **self.model_config.kwargs
        )

    def extract_json_from_response(self, input_response: str) -> list[list[int]] | None:
        """Extract JSON specifically for OpenAI's potential response formats."""
        # This uses call_ai_model, which might be inefficient if just parsing existing text.
        # Consider refactoring if needed, but for now, keep original logic.
        prompt = f"""
You are a helpful assistant. Extract only the JSON array of arrays from the following response. 
Do not include any explanation, formatting, or additional text.
Return ONLY the valid JSON array of arrays with integers.

Response:
{input_response}

Example of expected output format:
[[1, 2, 3], [4, 5, 6]]

IMPORTANT: Return ONLY the array, with no additional text, quotes, or formatting.
"""
        completion = self.call_ai_model(prompt=prompt)
        assistant_content = self._get_content(completion)

        # Extraction logic (same as before)
        if "```" in assistant_content:
            code_blocks = assistant_content.split("```")
            for block in code_blocks:
                if block.strip() and not block.strip().startswith("json"):
                    assistant_content = block.strip()
                    break
        
        assistant_content = assistant_content.strip()
        
        if assistant_content and not assistant_content.startswith("["):
            start_idx = assistant_content.find("[[")
            if start_idx >= 0:
                end_idx = assistant_content.rfind("]]") + 2
                if end_idx > start_idx:
                    assistant_content = assistant_content[start_idx:end_idx]

        try:
            json_result = json.loads(assistant_content)
            if isinstance(json_result, list) and all(isinstance(item, list) for item in json_result):
                return json_result
            if isinstance(json_result, dict) and "response" in json_result:
                 json_response = json_result.get("response")
                 if isinstance(json_response, list) and all(isinstance(item, list) for item in json_response):
                    return json_response
            return None
        except json.JSONDecodeError:
            try:
                array_pattern = r'\[\s*\[\s*\d+(?:\s*,\s*\d+)*\s*\](?:\s*,\s*\[\s*\d+(?:\s*,\s*\d+)*\s*\])*\s*\]'
                match = re.search(array_pattern, assistant_content)
                if match:
                    parsed_match = json.loads(match.group(0))
                    if isinstance(parsed_match, list) and all(isinstance(item, list) for item in parsed_match):
                        return parsed_match
            except:
                pass # Fall through if regex or secondary parse fails
            return None
        
    def _get_usage(self, response: Any) -> Usage:
        """Extract usage information specifically from an OpenAI response object."""
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        reasoning_tokens = 0 # Default

        if hasattr(response, 'usage') and response.usage:
            if self.model_config.api_type == APIType.CHAT_COMPLETIONS:
                prompt_tokens = response.usage.prompt_tokens
                completion_tokens = response.usage.completion_tokens
                total_tokens = response.usage.total_tokens
                # Safely access potential reasoning tokens
                if hasattr(response.usage, 'completion_tokens_details') and response.usage.completion_tokens_details and hasattr(response.usage.completion_tokens_details, 'reasoning_tokens'):
                    reasoning_tokens = response.usage.completion_tokens_details.reasoning_tokens or 0
            
            else: # APIType.RESPONSES (Assume this structure if not CHAT_COMPLETIONS)
                prompt_tokens = response.usage.input_tokens
                completion_tokens = response.usage.output_tokens
                total_tokens = prompt_tokens + completion_tokens # Responses API doesn't always return total
                # Safely access potential reasoning tokens
                if hasattr(response.usage, 'output_tokens_details') and response.usage.output_tokens_details and hasattr(response.usage.output_tokens_details, 'reasoning_tokens'):
                    reasoning_tokens = response.usage.output_tokens_details.reasoning_tokens or 0
        else:
            # Handle cases where usage might be missing (should log this)
            print(f"Warning: Usage information missing in response for model {self.model_config.model_name}") 

        return Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            completion_tokens_details=CompletionTokensDetails(
                reasoning_tokens=reasoning_tokens,
                accepted_prediction_tokens=completion_tokens, # Assuming all completion tokens are accepted for now
                rejected_prediction_tokens=0 # Assuming none rejected for now
            )
        )

    def _get_content(self, response: Any) -> str:
        """Extract content specifically from an OpenAI response object."""
        content = ""
        if self.model_config.api_type == APIType.CHAT_COMPLETIONS:
            if response.choices and response.choices[0].message:
                content = response.choices[0].message.content or ""
        else: # APIType.RESPONSES
            if hasattr(response, 'output_text'):
                 content = response.output_text or ""
        return content.strip()

    def _get_role(self, response: Any) -> str:
        """Extract role specifically from an OpenAI response object."""
        if self.model_config.api_type == APIType.CHAT_COMPLETIONS:
            if response.choices and response.choices[0].message:
                 return response.choices[0].message.role or "assistant"
        # Responses API always returns assistant content
        return "assistant"

    def _normalize_to_responses_kwargs(self):
        """
        Normalize kwargs based on API type to handle different parameter names between chat completions and responses APIs
        """
        if self.model_config.api_type == APIType.RESPONSES:
            # Convert max_tokens and max_completion_tokens to max_output_tokens for responses API
            if "max_tokens" in self.model_config.kwargs:
                self.model_config.kwargs["max_output_tokens"] = self.model_config.kwargs.pop("max_tokens")
            if "max_completion_tokens" in self.model_config.kwargs:
                self.model_config.kwargs["max_output_tokens"] = self.model_config.kwargs.pop("max_completion_tokens")