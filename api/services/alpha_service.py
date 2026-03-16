"""
AlphaService: LLM-generated alpha factors with JSONL cache and deterministic fallback.

Generates 8 numeric alpha factors per (asset_id, date) via structured LLM prompts.
Results are cached in JSONL format for offline training consumption.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def build_alpha_prompt(
    asset_id: str,
    as_of_date: str,
    feature_snapshot: dict,
    sentiment_snapshot: dict | None = None,
) -> str:
    """Construct a structured prompt instructing the LLM to output 8 numeric factors.

    Args:
        asset_id: Ticker symbol.
        as_of_date: Date string (YYYY-MM-DD).
        feature_snapshot: Dict of recent feature values for context.
        sentiment_snapshot: Optional dict of sentiment summary values.

    Returns:
        Prompt string for the LLM.
    """
    sentiment_block = ''
    if sentiment_snapshot:
        sentiment_block = f"""
Sentiment summary:
- Mean sentiment: {sentiment_snapshot.get('sentiment_mean', 0.0):.4f}
- Sentiment std: {sentiment_snapshot.get('sentiment_std', 0.0):.4f}
- Positive ratio: {sentiment_snapshot.get('sentiment_pos_ratio', 0.5):.4f}
- Article count: {sentiment_snapshot.get('sentiment_count', 0)}
"""
    feature_lines = '\n'.join(f'  {k}: {v}' for k, v in feature_snapshot.items())
    return f"""You are a quantitative analyst. Given the following market data for {asset_id} as of {as_of_date}, generate exactly 8 numeric alpha factors.

Recent features:
{feature_lines}
{sentiment_block}
Output ONLY a JSON object with keys "alpha_1" through "alpha_8", each a float in [-3, 3].
Example: {{"alpha_1": 0.5, "alpha_2": -1.2, "alpha_3": 0.0, "alpha_4": 0.8, "alpha_5": -0.3, "alpha_6": 1.1, "alpha_7": -0.7, "alpha_8": 0.2}}
"""


def deterministic_fallback_alphas(feature_snapshot: dict) -> dict[str, float]:
    """Compute simple deterministic baseline alphas when LLM is unavailable.

    Uses a hash-based approach seeded by feature values to produce
    reproducible float values in [-1, 1].

    Args:
        feature_snapshot: Dict of recent feature values.

    Returns:
        Dict with keys alpha_1 through alpha_8, all floats.
    """
    import hashlib

    raw = json.dumps(feature_snapshot, sort_keys=True, default=str)
    digest = hashlib.sha256(raw.encode()).hexdigest()
    alphas: dict[str, float] = {}
    for i in range(8):
        # Extract 4 hex chars per alpha → value in [0, 65535] → map to [-1, 1]
        hex_slice = digest[i * 4 : (i + 1) * 4]
        val = int(hex_slice, 16) / 65535.0 * 2.0 - 1.0
        alphas[f'alpha_{i + 1}'] = round(val, 6)
    return alphas


class AlphaService:
    """LLM-based alpha factor generation with JSONL caching.

    Args:
        model_name: OpenAI model identifier (e.g., 'gpt-4.1-mini').
        cache_path: Path to JSONL cache file.
        temperature: LLM sampling temperature.
    """

    def __init__(
        self,
        model_name: str = 'gpt-4.1-mini',
        cache_path: str = 'artifacts/data/phase3/alpha_cache.jsonl',
        temperature: float = 0.1,
    ):
        self.model_name = model_name
        self.cache_path = Path(cache_path)
        self.temperature = temperature
        self._client = None

    def _get_client(self):
        """Lazily initialise the OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI

                self._client = OpenAI()
            except Exception as e:
                logger.warning('Failed to initialise OpenAI client: %s', e)
                raise
        return self._client

    def get_cached_alphas(self, asset_id: str, as_of_date: str) -> dict | None:
        """Look up JSONL cache by (asset_id, as_of_date) key.

        Args:
            asset_id: Ticker symbol.
            as_of_date: Date string (YYYY-MM-DD).

        Returns:
            Cached alpha dict or None if not found.
        """
        if not self.cache_path.exists():
            return None
        cache_key = f'{asset_id}_{as_of_date}'
        try:
            with open(self.cache_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if entry.get('cache_key') == cache_key:
                        return entry
        except OSError:
            logger.warning('Could not read cache file %s', self.cache_path)
        return None

    def save_cached_alphas(self, payload: dict) -> None:
        """Append a single alpha result to the JSONL cache file.

        Args:
            payload: Dict containing cache_key, alpha values, and metadata.
        """
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.cache_path, 'a') as f:
                f.write(json.dumps(payload) + '\n')
        except OSError:
            logger.warning('Could not write to cache file %s', self.cache_path)

    def generate_alphas(
        self,
        asset_id: str,
        as_of_date: str,
        feature_snapshot: dict,
        sentiment_snapshot: dict | None = None,
    ) -> dict:
        """Generate alpha factors via LLM, with cache and deterministic fallback.

        Args:
            asset_id: Ticker symbol.
            as_of_date: Date string (YYYY-MM-DD).
            feature_snapshot: Dict of recent feature values.
            sentiment_snapshot: Optional sentiment summary dict.

        Returns:
            Dict with alpha_1..alpha_8, rationale, cache_key, cached flag.
        """
        cache_key = f'{asset_id}_{as_of_date}'

        # Check cache first
        cached = self.get_cached_alphas(asset_id, as_of_date)
        if cached is not None:
            cached['cached'] = True
            return cached

        # Try LLM generation
        prompt = build_alpha_prompt(
            asset_id, as_of_date, feature_snapshot, sentiment_snapshot
        )
        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[{'role': 'user', 'content': prompt}],
                temperature=self.temperature,
                max_tokens=256,
            )
            content = response.choices[0].message.content or ''
            # Parse JSON from response
            alpha_dict = json.loads(content)
            # Validate and clip to [-3, 3]
            alphas: dict[str, float] = {}
            for i in range(1, 9):
                key = f'alpha_{i}'
                val = float(alpha_dict.get(key, 0.0))
                alphas[key] = max(-3.0, min(3.0, val))
        except Exception:
            logger.warning(
                'LLM alpha generation failed for %s/%s; using deterministic fallback',
                asset_id,
                as_of_date,
                exc_info=True,
            )
            alphas = deterministic_fallback_alphas(feature_snapshot)

        result = {
            'cache_key': cache_key,
            'asset_id': asset_id,
            'date': as_of_date,
            **alphas,
            'rationale': None,
            'generated_at': datetime.utcnow().isoformat(),
            'cached': False,
        }
        self.save_cached_alphas(result)
        return result
