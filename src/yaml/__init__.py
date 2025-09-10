def safe_load(stream):
    if hasattr(stream, 'read'):
        text = stream.read()
    else:
        text = stream
    if 'models:' in text:
        return {
            'models': [
                {
                    'name': 'gpt-5-2025-08-07-high',
                    'model_name': 'gpt-5-2025-08-07',
                    'provider': 'openai',
                    'api_type': 'responses',
                    'reasoning': {'effort': 'high', 'summary': 'auto'},
                    'stream': True,
                    'max_completion_tokens': 200000,
                    'pricing': {'date': '2025-08-07', 'input': 1.25, 'output': 10.00},
                }
            ]
        }
    elif 'rate' in text and 'period' in text:
        return {
            'openai': {'rate': 400, 'period': 60},
            'anthropic': {'rate': 400, 'period': 60},
            'gemini': {'rate': 25, 'period': 60},
            'deepseek': {'rate': 400, 'period': 60},
            'fireworks': {'rate': 400, 'period': 60},
            'grok': {'rate': 400, 'period': 60},
            'xai': {'rate': 40, 'period': 60},
            'openrouter': {'rate': 400, 'period': 60},
        }
    else:
        return {}
