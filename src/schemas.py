from pydantic import BaseModel, model_validator, root_validator
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
import json

class APIType:
    """
    Enum for the different API types that can be used with the OpenAI API.
    """
    CHAT_COMPLETIONS = "chat_completions"
    RESPONSES = "responses"

class ARCTaskOutput(BaseModel):
    output: List[List[int]]

class ARCPair(BaseModel):
    input: List[List[int]]
    output: Optional[List[List[int]]] = None

class ARCTask(BaseModel):
    train: List[ARCPair]
    test: List[ARCPair]

class Message(BaseModel):
    role: str
    content: str

class Choice(BaseModel):
    index: int
    message: Message

class CompletionTokensDetails(BaseModel):
    reasoning_tokens: int
    accepted_prediction_tokens: int
    rejected_prediction_tokens: int

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    completion_tokens_details: CompletionTokensDetails

class Cost(BaseModel):
    prompt_cost: float
    completion_cost: float # Cost of completion_tokens * output_cost_per_token
    reasoning_cost: Optional[float] = None  # Cost of reasoning_tokens * output_cost_per_token. Optional as not all providers return it.
    total_cost: float      # Sum of prompt_cost, completion_cost, and reasoning_cost

class AttemptMetadata(BaseModel):
    model: str
    provider: str
    start_timestamp: datetime
    end_timestamp: datetime
    choices: List[Choice]
    reasoning_summary: Optional[str] = None
    kwargs: Dict[str, Any]
    usage: Usage
    cost: Cost
    task_id: Optional[str] = None
    pair_index: Optional[int] = None
    test_id: Optional[str] = None
    
    model_config = {
        'json_encoders': {
            datetime: lambda v: v.isoformat()
        }
    }

class Attempt(BaseModel):
    answer: Union[str, List[List[int]]]
    metadata: AttemptMetadata
    
    @model_validator(mode='before')
    @classmethod
    def validate_answer(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure answer is properly serialized"""
        if not isinstance(values, dict):
            return values
            
        answer = values.get('answer')
        if isinstance(answer, list):
            # Convert nested list to string representation
            values['answer'] = json.dumps(answer)
            
        return values
    
    model_config = {
        'json_encoders': {
            datetime: lambda v: v.isoformat()
        }
    }

class Attempts(BaseModel):
    attempts: List[Attempt]

class ModelPricing(BaseModel):
    date: str
    input: float
    output: float

class ModelConfig(BaseModel):
    """
    A model configuration used to populate a models kwargs and calculate pricing metadata. Not all fields are required by all providers.
    Points to model.yml
    """
    name: str  # This is now the config_name
    model_name: str  # The actual model name to use with the provider's API
    provider: str
    pricing: ModelPricing
    api_type: Optional[str] = APIType.CHAT_COMPLETIONS # currently only used by openai
    kwargs: Dict[str, Any] = {}
    
    model_config = {
        'protected_namespaces': (),
        'extra': 'allow'
    }
    
    @model_validator(mode='before')
    @classmethod
    def extract_kwargs(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Extract all extra fields into kwargs"""
        if not isinstance(values, dict):
            return values
            
        kwargs = {}
        known_fields = {'name', 'provider', 'pricing', 'kwargs', 'model_name', 'api_type'}
        
        for field_name, value in values.items():
            if field_name not in known_fields:
                kwargs[field_name] = value
                
        # Update the kwargs field with our extracted values
        if kwargs:
            values['kwargs'] = {**kwargs, **values.get('kwargs', {})}
            
            # Remove the extracted fields from the top level
            for field_name in kwargs:
                if field_name in values:
                    del values[field_name]
                    
        return values