import abc
from typing import List, Dict, Tuple, Any
import json
from datetime import datetime
from src.schemas import Attempt, ModelConfig
from src.utils.task_utils import read_models_config

class ProviderAdapter(abc.ABC):
    def __init__(self, model_name: str):
        """
        Initialize the provider adapter with model configuration
        
        Args:
            model_name: Name of the model to use
        """
        self.model_name = model_name
        self.model_config = read_models_config(model_name)
        
        # Verify the provider matches the adapter
        adapter_provider = self.__class__.__name__.lower().replace('adapter', '')
        if adapter_provider != self.model_config.provider:
            raise ValueError(f"Model provider mismatch. Model '{model_name}' is for provider '{self.model_config.provider}' but was passed to {self.__class__.__name__}")
        
        # Initialize the client
        self.client = self.init_client()

    @abc.abstractmethod
    def init_client(self):
        """
        Initialize the client for the provider. Each adapter must implement this.
        Should handle API key validation and client setup.
        """
        pass
    
    @abc.abstractmethod
    def make_prediction(self, prompt: str) -> Attempt:
        """
        Make a prediction with the model and return an Attempt object's answer
        """
        pass

    @abc.abstractmethod
    def chat_completion(self, messages: List[Dict[str, str]], tools: List[Dict[str, Any]] = []) -> Any:
        """
        Make a raw API call to the provider and return the response
        """
        pass

    @abc.abstractmethod
    def extract_json_from_response(self, input_response: str) -> List[List[int]]:
        """
        Extract JSON from various possible formats in the response
        """
        pass