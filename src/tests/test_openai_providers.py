import pytest
from unittest.mock import Mock, MagicMock, patch
import inspect # Added for subclass discovery
import sys # Added for subclass discovery

from src.adapters.openai_base import OpenAIBaseAdapter
from src.adapters.open_ai import OpenAIAdapter
from src.adapters.grok import GrokAdapter
from src.adapters.deepseek import DeepseekAdapter
from src.adapters.fireworks import FireworksAdapter
# Import all adapters to ensure they are available for discovery
import src.adapters 

from src.schemas import ModelConfig, ModelPricing, Usage, CompletionTokensDetails, Cost, APIType
from src.errors import TokenMismatchError
import os
from dotenv import load_dotenv

load_dotenv() # Load environment variables for potential API key checks during init

# --- Helper Function for Subclass Discovery ---
def find_concrete_subclasses(cls):
    """Finds all concrete subclasses of a given class currently imported."""
    concrete_subclasses = []
    for name, obj in inspect.getmembers(sys.modules[cls.__module__.split('.')[0]]): # Look within the 'src' module tree
        if inspect.ismodule(obj):
             for sub_name, sub_obj in inspect.getmembers(obj):
                  if inspect.isclass(sub_obj) and \
                     issubclass(sub_obj, cls) and \
                     sub_obj != cls and \
                     not inspect.isabstract(sub_obj):
                      # Check specifically within adapters submodule
                      if 'adapters' in sub_obj.__module__:
                           concrete_subclasses.append(sub_obj)
                           
    # Need to also check adapters imported directly
    for name, obj in inspect.getmembers(src.adapters):
         if inspect.isclass(obj) and issubclass(obj, cls) and obj != cls and not inspect.isabstract(obj):
             if obj not in concrete_subclasses:
                  concrete_subclasses.append(obj)
                  
    return concrete_subclasses

# Dynamically find all concrete subclasses of OpenAIBaseAdapter
openai_adapter_subclasses = find_concrete_subclasses(OpenAIBaseAdapter)

# --- Fixtures ---

@pytest.fixture
def mock_model_config(adapter_class):
    """Provides a mock ModelConfig, adapting provider based on the adapter class."""
    # Infer provider name from class name (e.g., OpenAIAdapter -> openai)
    provider_name = adapter_class.__name__.lower().replace("adapter", "")
    config_name = f"test-{provider_name}-model"
    model_name = f"{provider_name}-test-model"
    
    # Handle potential variations if needed (e.g., fireworks vs fireworks-ai)
    # For now, simple lowercase name works for openai, grok, deepseek, fireworks

    return ModelConfig(
        name=config_name,
        model_name=model_name,
        provider=provider_name, 
        pricing=ModelPricing(date="2024-01-01", input=1.0, output=2.0), 
        api_type=APIType.CHAT_COMPLETIONS, # Assuming chat completions for simplicity
        kwargs={"temperature": 0.5}
    )

@pytest.fixture
def mock_openai_response_usage():
    """Provides a mock OpenAI API response object with usage data."""
    mock_response = Mock()
    mock_response.usage = Mock()
    mock_response.usage.prompt_tokens = 100
    mock_response.usage.completion_tokens = 200
    mock_response.usage.total_tokens = 300
    # Mock completion_tokens_details (assuming not present initially)
    mock_response.usage.completion_tokens_details = None
    return mock_response
    
@pytest.fixture
def mock_openai_response_reasoning():
    """Provides a mock OpenAI API response with reasoning tokens."""
    mock_response = Mock()
    mock_response.usage = Mock()
    mock_response.usage.prompt_tokens = 50
    mock_response.usage.completion_tokens = 150
    mock_response.usage.total_tokens = 210 # Includes 10 reasoning tokens
    mock_response.usage.completion_tokens_details = Mock()
    mock_response.usage.completion_tokens_details.reasoning_tokens = 10
    return mock_response

@pytest.fixture
def mock_openai_response_mismatch():
    """Provides a mock OpenAI API response with mismatched token counts that cannot be reconciled."""
    mock_response = Mock()
    mock_response.usage = Mock()
    mock_response.usage.prompt_tokens = 100
    mock_response.usage.completion_tokens = 200
    mock_response.usage.total_tokens = 350  # Provider claims 350
    # Explicitly claims some reasoning tokens, but not enough to reconcile
    mock_response.usage.completion_tokens_details = Mock()
    mock_response.usage.completion_tokens_details.reasoning_tokens = 10  # Sum becomes 310, still mismatch
    return mock_response
    
@pytest.fixture
def mock_openai_response_infer_reasoning():
    """Provides a mock response where reasoning tokens should be inferred."""
    mock_response = Mock()
    mock_response.usage = Mock()
    mock_response.usage.prompt_tokens = 100
    mock_response.usage.completion_tokens = 200
    mock_response.usage.total_tokens = 320 # Implies 20 reasoning tokens
    # Simulate provider not giving the details object or reasoning_tokens=0
    mock_response.usage.completion_tokens_details = Mock()
    mock_response.usage.completion_tokens_details.reasoning_tokens = 0
    return mock_response


# --- Test Class ---
# Parameterize the whole class to run tests for each adapter subclass
@pytest.mark.parametrize("adapter_class", openai_adapter_subclasses)
class TestOpenAIBaseProviderLogic:

    @pytest.fixture(autouse=True)
    def patch_env_vars(self, monkeypatch):
        """Ensure necessary API keys are mocked for adapter initialization."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("XAI_API_KEY", "test-key")
        monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
        monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")

    @pytest.fixture
    def adapter_instance(self, adapter_class, mock_model_config):
        """Provides an instance of the parameterized adapter class, bypassing real init."""
        
        # Patch the base ProviderAdapter.__init__ to prevent it from running
        # We will manually set the necessary attributes (model_config, client)
        with patch("src.adapters.provider.ProviderAdapter.__init__", return_value=None) as mock_provider_init:
            
            # Instantiate the parameterized adapter class (init is mocked)
            adapter = adapter_class(config=mock_model_config.name) 
            
            # Manually set the attributes that ProviderAdapter.__init__ would normally set
            adapter.config = mock_model_config.name
            adapter.model_config = mock_model_config 
            adapter.client = MagicMock() # Mock the client instance directly
            
            return adapter

    def test_get_usage_standard(self, adapter_instance, mock_openai_response_usage):
        """Test _get_usage extracts tokens correctly from a standard response."""
        usage = adapter_instance._get_usage(mock_openai_response_usage)
        assert isinstance(usage, Usage)
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 200
        assert usage.total_tokens == 300
        assert usage.completion_tokens_details.reasoning_tokens == 0 # Default

    def test_get_usage_with_reasoning(self, adapter_instance, mock_openai_response_reasoning):
        """Test _get_usage extracts reasoning tokens when provided."""
        usage = adapter_instance._get_usage(mock_openai_response_reasoning)
        assert usage.prompt_tokens == 50
        assert usage.completion_tokens == 150
        assert usage.total_tokens == 210
        assert usage.completion_tokens_details.reasoning_tokens == 10

    def test_calculate_output_cost_standard(self, adapter_instance, mock_openai_response_usage):
        """Test _calculate_output_cost calculates cost correctly."""
        cost = adapter_instance._calculate_output_cost(mock_openai_response_usage)
        expected_prompt_cost = 100 * (1.0 / 1_000_000)
        expected_completion_cost = 200 * (2.0 / 1_000_000) # No reasoning tokens
        expected_total_cost = expected_prompt_cost + expected_completion_cost
        
        assert isinstance(cost, Cost)
        assert cost.prompt_cost == pytest.approx(expected_prompt_cost)
        assert cost.completion_cost == pytest.approx(expected_completion_cost)
        assert cost.total_cost == pytest.approx(expected_total_cost)

    def test_calculate_output_cost_with_reasoning(self, adapter_instance, mock_openai_response_reasoning):
        """Test _calculate_output_cost includes reasoning tokens in output cost."""
        cost = adapter_instance._calculate_output_cost(mock_openai_response_reasoning)
        expected_prompt_cost = 50 * (1.0 / 1_000_000)
         # Output cost includes completion + reasoning tokens
        expected_completion_cost = 150 * (2.0 / 1_000_000)  # Based only on completion tokens
        expected_reasoning_cost = 10 * (2.0 / 1_000_000)   # Based only on reasoning tokens
        expected_total_cost = expected_prompt_cost + expected_completion_cost

        assert cost.prompt_cost == pytest.approx(expected_prompt_cost)
        assert cost.completion_cost == pytest.approx(expected_completion_cost)
        assert cost.reasoning_cost == pytest.approx(expected_reasoning_cost)
        assert cost.total_cost == pytest.approx(expected_total_cost)
        
    def test_calculate_output_cost_token_mismatch_error(self, adapter_instance, mock_openai_response_mismatch):
        """Test _calculate_output_cost raises TokenMismatchError on mismatch."""
        with pytest.raises(TokenMismatchError) as excinfo:
            adapter_instance._calculate_output_cost(mock_openai_response_mismatch)
        # Check the new error message format
        expected_msg_part1 = "Token count mismatch: API reports total 350"
        # Updated expected message based on latest code change
        expected_msg_part2 = "but computed P:100 + C:200 + R:10 = 310"
        assert expected_msg_part1 in str(excinfo.value)
        assert expected_msg_part2 in str(excinfo.value)

    # --- Test make_prediction flow for each subclass ---
    # Patching needs to target the specific adapter_class being tested
    def test_make_prediction_flow(self, adapter_class, adapter_instance, mock_openai_response_usage):
        """Test the overall flow of make_prediction in the parameterized subclass."""
        
        # Patch the methods on the *specific class* being tested
        with patch.object(adapter_class, '_call_ai_model') as mock_call_ai, \
             patch.object(adapter_class, '_get_content', return_value="Test Answer") as mock_get_content, \
             patch.object(adapter_class, '_get_role', return_value="assistant") as mock_get_role:

            mock_call_ai.return_value = mock_openai_response_usage # Make the mocked call return our response
            
            prompt = "Test prompt"
            attempt = adapter_instance.make_prediction(prompt)

            mock_call_ai.assert_called_once_with(prompt)
            # We expect _get_content and _get_role to be called, 
            # potentially multiple times depending on implementation, check they were called at least once.
            assert mock_get_content.called
            assert mock_get_role.called
            
            # Check the final attempt object
            assert attempt.answer == "Test Answer"
            assert attempt.metadata.model == adapter_instance.model_config.model_name
            assert attempt.metadata.provider == adapter_instance.model_config.provider
            assert attempt.metadata.usage.prompt_tokens == 100
            assert attempt.metadata.usage.completion_tokens == 200
            assert attempt.metadata.usage.total_tokens == 300
            # Reasoning tokens might be 0 or inferred (0 in this mock response case)
            assert attempt.metadata.usage.completion_tokens_details.reasoning_tokens == 0 
            
            expected_prompt_cost = 100 * (1.0 / 1_000_000)
            # Cost calculation depends on reasoning tokens, which are 0 here
            expected_completion_cost = 200 * (2.0 / 1_000_000)
            expected_reasoning_cost = 0 * (2.0 / 1_000_000)
            assert attempt.metadata.cost.prompt_cost == pytest.approx(expected_prompt_cost)
            assert attempt.metadata.cost.completion_cost == pytest.approx(expected_completion_cost)
            assert attempt.metadata.cost.reasoning_cost == pytest.approx(expected_reasoning_cost)
            assert attempt.metadata.cost.total_cost == pytest.approx(expected_prompt_cost + expected_completion_cost)
            assert attempt.metadata.kwargs == {"temperature": 0.5}
            assert len(attempt.metadata.choices) == 2
            assert attempt.metadata.choices[0].message.role == "user"
            assert attempt.metadata.choices[0].message.content == prompt
            assert attempt.metadata.choices[1].message.role == "assistant"
            assert attempt.metadata.choices[1].message.content == "Test Answer"

# You could add similar tests specifically targeting GrokAdapter, DeepseekAdapter etc.
# or parameterize the tests if the setup is sufficiently similar.
# For now, this covers the core logic inherited from OpenAIBaseAdapter. 