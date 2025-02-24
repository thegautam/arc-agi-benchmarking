import abc
from typing import List, Dict, Tuple, Any
import json
from datetime import datetime
from src.schemas import Attempt

class ProviderAdapter(abc.ABC):

    @abc.abstractmethod
    def init_client(self):
        """
        Initialize the client for the provider
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