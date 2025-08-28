from copy import deepcopy
from datetime import datetime, timezone
from typing import Optional

from arc_agi_benchmarking.cache.provider_cache import ProviderCache
from arc_agi_benchmarking.schemas import Attempt, ModelConfig, AttemptMetadata, Usage, Cost


class CachedAdapter:
    """
    A lightweight wrapper that adds caching to any provider adapter exposing:
      - model_config: ModelConfig
      - make_prediction(prompt, task_id=None, test_id=None, pair_index=None) -> Attempt
      - extract_json_from_response(text) (delegated)
    """

    def __init__(
        self,
        adapter,
        cache_dir: Optional[str] = None,
        enabled: bool = True,
        zero_cost_on_hit: bool = True,
    ) -> None:
        self._adapter = adapter
        self._cache = ProviderCache(cache_dir)
        self._enabled = enabled
        self._zero_cost_on_hit = zero_cost_on_hit

    @property
    def model_config(self) -> ModelConfig:
        return self._adapter.model_config

    def extract_json_from_response(self, input_response: str):
        return self._adapter.extract_json_from_response(input_response)

    def make_prediction(self, prompt: str, task_id: Optional[str] = None, test_id: Optional[str] = None, pair_index: int = None) -> Attempt:
        if not self._enabled:
            return self._adapter.make_prediction(prompt, task_id=task_id, test_id=test_id, pair_index=pair_index)

        key = ProviderCache.build_key_dict(prompt=prompt, model_config=self.model_config)
        hit = self._cache.get(key)
        if hit is not None:
            # Return a copy adjusted for current call context
            now = datetime.now(timezone.utc)
            md = deepcopy(hit.metadata)
            md.task_id = task_id
            md.test_id = test_id
            md.pair_index = pair_index
            md.start_timestamp = now
            md.end_timestamp = now
            if self._zero_cost_on_hit:
                md.usage = Usage(
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    completion_tokens_details=md.usage.completion_tokens_details.__class__(
                        reasoning_tokens=0,
                        accepted_prediction_tokens=0,
                        rejected_prediction_tokens=0,
                    ),
                )
                md.cost = Cost(prompt_cost=0.0, completion_cost=0.0, reasoning_cost=0.0, total_cost=0.0)
            attempt = Attempt(answer=hit.answer, code=hit.code, metadata=md, correct=hit.correct)
            return attempt

        # Miss: call provider and store
        attempt = self._adapter.make_prediction(prompt, task_id=task_id, test_id=test_id, pair_index=pair_index)
        try:
            self._cache.set(key, attempt)
        except Exception:
            # Don't block on cache write failures
            pass
        return attempt
