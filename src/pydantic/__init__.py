class BaseModel:
    def __init__(self, **data):
        annotations = getattr(self, '__annotations__', {})
        for name, field_type in annotations.items():
            if name in data:
                value = data[name]
                if isinstance(field_type, type) and issubclass(field_type, BaseModel) and isinstance(value, dict):
                    value = field_type(**value)
                setattr(self, name, value)
            else:
                if hasattr(self, name):
                    setattr(self, name, getattr(self, name))
                else:
                    setattr(self, name, None)
        for name, value in data.items():
            if name not in annotations:
                setattr(self, name, value)

    def model_dump(self):
        result = {}
        annotations = getattr(self, '__annotations__', {})
        for name in annotations:
            value = getattr(self, name, None)
            if isinstance(value, BaseModel):
                result[name] = value.model_dump()
            else:
                result[name] = value
        for name, value in self.__dict__.items():
            if name not in result:
                if isinstance(value, BaseModel):
                    result[name] = value.model_dump()
                else:
                    result[name] = value
        return result

    @classmethod
    def model_validate(cls, data):
        if isinstance(data, cls):
            return data
        return cls(**data)

def model_validator(*args, **kwargs):
    def decorator(func):
        return func
    return decorator

class ValidationError(Exception):
    pass
