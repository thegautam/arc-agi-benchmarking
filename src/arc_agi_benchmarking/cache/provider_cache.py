import os
import json
import hashlib
from typing import Optional, Dict, Any, List
from pydantic import ValidationError

from arc_agi_benchmarking.schemas import Attempt, ModelConfig, APIType


def _stable_json_dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


class ProviderCache:
    """
    Simple file-system cache for provider attempts.
    Keys are constructed from prompt + model configuration parameters that affect output.
    Values are serialized Attempt objects (JSON).
    """

    def __init__(self, cache_dir: Optional[str] = None) -> None:
        base_dir = cache_dir or os.getenv("ARC_AGI_CACHE_DIR", ".cache")
        self.cache_dir = base_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    @staticmethod
    def _canonicalize_kwargs(api_type: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Return a canonicalized copy of kwargs for use in cache keys.
        In particular, normalize token limit parameter names for the Responses API
        so keys remain stable regardless of whether callers use max_tokens,
        max_completion_tokens, or max_output_tokens.
        """
        if not kwargs:
            return {}
        canon = dict(kwargs)
        if api_type == APIType.RESPONSES:
            # Prefer explicit max_output_tokens if provided
            if "max_output_tokens" in canon:
                # Remove synonyms to avoid key differences
                canon.pop("max_tokens", None)
                canon.pop("max_completion_tokens", None)
            else:
                # Map common synonyms to max_output_tokens
                if "max_completion_tokens" in canon:
                    canon["max_output_tokens"] = canon.pop("max_completion_tokens")
                    canon.pop("max_tokens", None)
                elif "max_tokens" in canon:
                    canon["max_output_tokens"] = canon.pop("max_tokens")
                    canon.pop("max_completion_tokens", None)
        return canon

    @staticmethod
    def build_key_dict(prompt: Optional[str], model_config: ModelConfig, messages: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Build a canonical key dict. Only include parameters that influence model output.
        Excludes run-specific metadata (task_id, test_id, pair_index, timestamps).
        If a messages list is provided, use it instead of a single prompt string so different
        conversation states produce distinct keys.
        """
        canonical_kwargs = ProviderCache._canonicalize_kwargs(model_config.api_type, model_config.kwargs or {})
        key: Dict[str, Any] = {
            "provider": model_config.provider,
            "model_name": model_config.model_name,
            "api_type": model_config.api_type,
            "kwargs": canonical_kwargs,
        }
        if messages is not None:
            # Normalize messages to role/content pairs only for stability
            key["messages"] = [
                {"role": m.get("role"), "content": m.get("content")}
                for m in messages
                if isinstance(m, dict)
            ]
        else:
            key["prompt"] = prompt or ""
        return key

    @staticmethod
    def key_to_hash(key_dict: Dict[str, Any]) -> str:
        canonical = _stable_json_dumps(key_dict)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _path_for_key(self, key_hash: str) -> str:
        # Use subdirectories to avoid too many files per directory
        prefix = key_hash[:2]
        d = os.path.join(self.cache_dir, prefix)
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, f"{key_hash}.json")

    def get(self, key_dict: Dict[str, Any]) -> Optional[Attempt]:
        key_hash = self.key_to_hash(key_dict)
        path = self._path_for_key(key_hash)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return Attempt.model_validate(data)
        except (json.JSONDecodeError, OSError, ValidationError):
            return None

    def set(self, key_dict: Dict[str, Any], attempt: Attempt) -> None:
        key_hash = self.key_to_hash(key_dict)
        path = self._path_for_key(key_hash)
        tmp_path = f"{path}.tmp"
        payload = attempt.model_dump(mode="json")
        # include minimal cache metadata
        payload["_cache_meta"] = {
            "key": key_dict,
            "version": 1,
        }
        data = _stable_json_dumps(payload)
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(data)
        # Atomic replace
        os.replace(tmp_path, path)
