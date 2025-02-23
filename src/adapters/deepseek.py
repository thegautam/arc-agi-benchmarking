from .provider import ProviderAdapter
import os
from dotenv import load_dotenv
import json
from openai import OpenAI
from datetime import datetime
from src.models import ARCTaskOutput, AttemptMetadata, Choice, Message, Usage, Cost, CompletionTokensDetails, Attempt
import logging

load_dotenv()
logger = logging.getLogger(__name__)

class DeepseekAdapter(ProviderAdapter):
    def __init__(self, model_name: str, max_tokens: int = 4024):
        logger.debug(f"Initializing DeepseekAdapter with model: {model_name}")
        self.client = self.init_client()
        self.model_name = model_name
        self.max_tokens = max_tokens
    
    def init_client(self):
        if not os.environ.get("DEEPSEEK_API_KEY"):
            raise ValueError("DEEPSEEK_API_KEY not found in environment variables")
        
        client = OpenAI(api_key = os.environ.get("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
        return client

    def make_prediction(self, prompt: str) -> Attempt:
        start_time = datetime.utcnow()
        
        messages = [
            {"role": "user", "content": prompt}
        ]
        response = self.chat_completion(messages)
        
        end_time = datetime.utcnow()

        # Calculate costs based on Deepseek's pricing
        input_cost_per_token = 0.0000002  # $0.0002/1K tokens for Deepseek-Chat
        output_cost_per_token = 0.0000002  # $0.0002/1K tokens for Deepseek-Chat
        
        prompt_cost = response.usage.prompt_tokens * input_cost_per_token
        completion_cost = response.usage.completion_tokens * output_cost_per_token

        # Convert input messages to choices
        input_choices = [
            Choice(
                index=i,
                message=Message(
                    role=msg["role"],
                    content=msg["content"]
                )
            )
            for i, msg in enumerate(messages)
        ]

        # Convert Deepseek response to our schema
        response_choices = [
            Choice(
                index=len(input_choices),
                message=Message(
                    role=response.choices[0].message.role,
                    content=response.choices[0].message.content.strip()
                )
            )
        ]

        # Combine input and response choices
        all_choices = input_choices + response_choices

        # Create metadata using our Pydantic models
        metadata = AttemptMetadata(
            model=self.model_name,
            provider="deepseek",
            start_timestamp=start_time,
            end_timestamp=end_time,
            choices=all_choices,
            kwargs={
                "max_tokens": self.max_tokens,
            },
            usage=Usage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
                completion_tokens_details=CompletionTokensDetails(
                    reasoning_tokens=0,  # Deepseek doesn't provide this breakdown
                    accepted_prediction_tokens=response.usage.completion_tokens,
                    rejected_prediction_tokens=0  # Deepseek doesn't provide this
                )
            ),
            cost=Cost(
                prompt_cost=prompt_cost,
                completion_cost=completion_cost,
                total_cost=prompt_cost + completion_cost
            )
        )

        attempt = Attempt(
            metadata=metadata,
            answer=response.choices[0].message.content.strip()
        )

        return attempt

    def chat_completion(self, messages: str) -> str:
        # print(messages)
        return self.client.chat.completions.create(
            model=self.model_name,
            # TODO: parameterize the reasoning_effort (including not setting it since it's only supported
            # o1, as of 12/19/2024)
            # Default value for o1 is 'medium'.
            # Uncomment to set a different value.
            # reasoning_effort='high',
            messages=messages,
        )

    def extract_json_from_response(self, input_response: str) -> list[list[int]] | None:


        prompt = f"""
You are a helpful assistant. Extract only the JSON of the test output from the following response. 
Do not include any explanation or additional text; only return valid JSON.

Response:
{input_response}

The JSON should be in this format:
{{
"response": [
    [1, 2, 3],
    [4, 5, 6]
]
}}
"""

        # print(f"Input response: {input_response}")
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        # Uncomment to print token usage 
        # print(f"USAGE|PROMPT|{completion.usage.prompt_tokens}")
        # print(f"USAGE|COMPLETION|{completion.usage.completion_tokens}")

        assistant_content = completion.choices[0].message.content.strip()

        # Some oai models like to wrap the response in a code block
        if assistant_content.startswith("```json"):
            assistant_content = "\n".join(assistant_content.split("\n")[1:])
        
        if assistant_content.endswith("```"):
            assistant_content = "\n".join(assistant_content.split("\n")[:-1])

        # Attempt to parse the returned content as JSON
        # print(f"For input response: {input_response}, got extracted content: {assistant_content}")
        try:
            json_entities = json.loads(assistant_content)
            return json_entities.get("response")
        except json.JSONDecodeError:
            return None