from pydantic import BaseModel, model_validator, root_validator
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
import json

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
    completion_cost: float
    total_cost: float

class AttemptMetadata(BaseModel):
    model: str
    provider: str
    start_timestamp: datetime
    end_timestamp: datetime
    choices: List[Choice]
    reasoning_tokens: Optional[str] = None
    kwargs: Dict[str, Any]
    usage: Usage
    cost: Cost
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
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
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class Attempts(BaseModel):
    attempts: List[Attempt]

class ModelPricing(BaseModel):
    date: str
    input: float
    output: float

class ModelConfig(BaseModel):
    """
    A model configuration used to populate a models kwargs and calculatepricing metadata. 
    Points to model.yml
    """
    name: str
    provider: str
    pricing: ModelPricing
    kwargs: Dict[str, Any] = {}

    @model_validator(mode='before')
    @classmethod
    def build_kwargs(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Collects any extra fields into kwargs"""
        if not isinstance(values, dict):
            return values
            
        kwargs = {}
        known_fields = {'name', 'provider', 'pricing', 'kwargs'}
        
        for field_name, value in values.items():
            if field_name not in known_fields:
                kwargs[field_name] = value
                
        if 'kwargs' in values:
            kwargs.update(values['kwargs'])
                
        values['kwargs'] = kwargs
        return values

    class Config:
        extra = "allow"  # Allow extra fields to be stored in kwargs