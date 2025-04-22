# This comment is just to trigger a rename operation via the edit tool
# The actual content change will happen in the next step. 

import abc
from .provider import ProviderAdapter
import os
from dotenv import load_dotenv
import json
from openai import OpenAI
from datetime import datetime, timezone
from src.schemas import APIType, AttemptMetadata, Choice, Message, Usage, Cost, CompletionTokensDetails, Attempt
from typing import Optional, Any, List, Dict # Added List, Dict

load_dotenv()


class OpenAIBaseAdapter(ProviderAdapter, abc.ABC):


    @abc.abstractmethod
    def make_prediction(self, prompt: str, task_id: Optional[str] = None, test_id: Optional[str] = None, pair_index: int = None) -> Attempt:
        """
        Make a prediction with the model and return an Attempt object.
        Subclasses must implement this to handle provider-specific response parsing.
        """
        pass

    def call_ai_model(self, prompt: str) -> Any:
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
    
    def chat_completion(self, messages: List[Dict[str, str]]) -> Any:
        """
        Make a call to the OpenAI Chat Completions API
        """
        return self.client.chat.completions.create(
            model=self.model_config.model_name,
            messages=messages,
            **self.model_config.kwargs
        )
    
    def responses(self, messages: List[Dict[str, str]]) -> Any:
        """
        Make a call to the OpenAI Responses API
        """
        return self.client.responses.create(
            model=self.model_config.model_name,
            input=messages,
            **self.model_config.kwargs
        )

    @abc.abstractmethod
    def extract_json_from_response(self, input_response: str) -> list[list[int]] | None:
        """
        Extract JSON from the provider's response.
        Subclasses must implement this based on expected response format.
        """
        pass
        
    @abc.abstractmethod
    def _get_usage(self, response: Any) -> Usage:
        """
        Extract usage information from the provider's response object.
        Subclasses must implement this based on the provider's response structure.
        """
        pass

    @abc.abstractmethod
    def _get_content(self, response: Any) -> str:
        """
        Extract the main content string from the provider's response object.
        Subclasses must implement this based on the provider's response structure.
        """
        pass

    @abc.abstractmethod
    def _get_role(self, response: Any) -> str:
        """
        Extract the role (e.g., 'assistant') from the provider's response object.
        Subclasses must implement this based on the provider's response structure.
        """
        pass
        
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