import time
from datetime import datetime, timezone
from typing import Optional, List

import pytest

from arc_agi_benchmarking.cache.cached_adapter import CachedAdapter
from arc_agi_benchmarking.cache.provider_cache import ProviderCache
from arc_agi_benchmarking.schemas import (
    Attempt,
    AttemptMetadata,
    Choice,
    Message,
    Usage,
    CompletionTokensDetails,
    Cost,
    ModelConfig,
    ModelPricing,
)


class DummyAdapter:
    """Minimal adapter with deterministic outputs for testing the cache wrapper."""
    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self.calls: List[str] = []

    def extract_json_from_response(self, input_response: str):
        return None

    def make_prediction(
        self,
        prompt: str,
        task_id: Optional[str] = None,
        test_id: Optional[str] = None,
        pair_index: int = None,
    ) -> Attempt:
        self.calls.append(prompt)
        now = datetime.now(timezone.utc)
        usage = Usage(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            completion_tokens_details=CompletionTokensDetails(
                reasoning_tokens=0,
                accepted_prediction_tokens=20,
                rejected_prediction_tokens=0,
            ),
        )
        cost = Cost(
            prompt_cost=0.001,
            completion_cost=0.002,
            reasoning_cost=0.0,
            total_cost=0.003,
        )
        md = AttemptMetadata(
            model=self.model_config.model_name,
            provider=self.model_config.provider,
            start_timestamp=now,
            end_timestamp=now,
            choices=[
                Choice(index=0, message=Message(role="user", content=prompt)),
                Choice(index=1, message=Message(role="assistant", content="[[1]]")),
            ],
            kwargs=self.model_config.kwargs,
            usage=usage,
            cost=cost,
            task_id=task_id,
            test_id=test_id,
            pair_index=pair_index,
        )
        return Attempt(answer=[[1]], code=None, metadata=md)


@pytest.fixture
def base_model_config() -> ModelConfig:
    return ModelConfig(
        name="test-config",
        model_name="test-model",
        provider="openai",
        pricing=ModelPricing(date="2024-01-01", input=1.0, output=2.0),
        kwargs={"temperature": 0.1},
    )


def test_cache_hit_and_miss(tmp_path, base_model_config):
    adapter = DummyAdapter(model_config=base_model_config)
    cached = CachedAdapter(adapter, cache_dir=str(tmp_path), enabled=True, zero_cost_on_hit=True)

    # First call -> miss (adapter called)
    a1 = cached.make_prediction("prompt-1", task_id="t1", test_id="cfgA", pair_index=0)
    assert adapter.calls == ["prompt-1"]
    assert a1.metadata.usage.total_tokens == 30
    assert a1.metadata.cost.total_cost == pytest.approx(0.003)

    # Second call with same key -> hit (adapter not called)
    time.sleep(0.01)  # ensure timestamps differ
    a2 = cached.make_prediction("prompt-1", task_id="t1b", test_id="cfgA", pair_index=1)
    assert adapter.calls == ["prompt-1"]  # no new call
    assert a2.answer == a1.answer
    # On hit, usage and cost should be zeroed (by default behavior)
    assert a2.metadata.usage.total_tokens == 0
    assert a2.metadata.usage.prompt_tokens == 0
    assert a2.metadata.usage.completion_tokens == 0
    assert a2.metadata.cost.total_cost == 0.0
    # Call-specific metadata should be updated
    assert a2.metadata.task_id == "t1b"
    assert a2.metadata.test_id == "cfgA"
    assert a2.metadata.pair_index == 1
    # Timestamps updated to now (not equal to original)
    assert a2.metadata.start_timestamp >= a1.metadata.start_timestamp
    assert a2.metadata.end_timestamp >= a2.metadata.start_timestamp

    # Different prompt -> miss
    a3 = cached.make_prediction("prompt-2", task_id="t2", test_id="cfgA", pair_index=0)
    assert adapter.calls == ["prompt-1", "prompt-2"]
    assert a3.metadata.usage.total_tokens == 30


def test_cache_key_includes_kwargs(tmp_path, base_model_config):
    # Same prompt but different kwargs should produce a miss
    adapter = DummyAdapter(model_config=base_model_config)
    cached = CachedAdapter(adapter, cache_dir=str(tmp_path), enabled=True)

    _ = cached.make_prediction("same-prompt")
    assert adapter.calls == ["same-prompt"]

    # Change a setting that affects output
    adapter.model_config.kwargs = {"temperature": 0.9}

    _ = cached.make_prediction("same-prompt")
    assert adapter.calls == ["same-prompt", "same-prompt"]


def test_cache_disabled_calls_underlying_every_time(tmp_path, base_model_config):
    adapter = DummyAdapter(model_config=base_model_config)
    cached = CachedAdapter(adapter, cache_dir=str(tmp_path), enabled=False)

    _ = cached.make_prediction("p")
    _ = cached.make_prediction("p")
    assert adapter.calls == ["p", "p"]


def test_zero_cost_on_hit_false_preserves_cost(tmp_path, base_model_config):
    adapter = DummyAdapter(model_config=base_model_config)
    cached = CachedAdapter(adapter, cache_dir=str(tmp_path), enabled=True, zero_cost_on_hit=False)

    a1 = cached.make_prediction("z")
    a2 = cached.make_prediction("z")

    # Second is a cache hit but should keep usage/cost from stored attempt
    assert a1.metadata.cost.total_cost == pytest.approx(0.003)
    assert a2.metadata.cost.total_cost == pytest.approx(0.003)
    assert a2.metadata.usage.total_tokens == 30


def test_provider_cache_hash_stability(tmp_path, base_model_config):
    prompt = "hash-test"
    key1 = ProviderCache.build_key_dict(prompt, base_model_config)
    key2 = ProviderCache.build_key_dict(prompt, base_model_config)
    assert key1 == key2
    h1 = ProviderCache.key_to_hash(key1)
    h2 = ProviderCache.key_to_hash(key2)
    assert h1 == h2
