"""Base fetcher with async HTTP, retry, rate-limiting, and file caching."""

import asyncio
import hashlib
import json
import logging
import time
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

DATA_RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


class BaseFetcher:
    """Async HTTP fetcher with rate limiting, retries, and local JSON caching."""

    def __init__(
        self,
        source_name: str,
        rate_limit_seconds: float = 0.1,
        max_retries: int = 3,
        headers: dict | None = None,
        cache_dir: str | None = None,
    ):
        self.source_name = source_name
        self.rate_limit_seconds = rate_limit_seconds
        self.max_retries = max_retries
        self.headers = headers or {}
        self.cache_dir = DATA_RAW_DIR / (cache_dir or source_name)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._last_request_time = 0.0
        self._semaphore = asyncio.Semaphore(1)

    async def _rate_limit(self):
        """Enforce minimum interval between requests."""
        async with self._semaphore:
            now = time.monotonic()
            elapsed = now - self._last_request_time
            if elapsed < self.rate_limit_seconds:
                await asyncio.sleep(self.rate_limit_seconds - elapsed)
            self._last_request_time = time.monotonic()

    def _cache_path(self, cache_key: str) -> Path:
        """Get the cache file path for a given key."""
        safe_key = cache_key.replace("/", "_").replace("\\", "_")
        return self.cache_dir / f"{safe_key}.json"

    def _read_cache(self, cache_key: str) -> dict | str | None:
        """Read cached response if it exists."""
        path = self._cache_path(cache_key)
        if path.exists():
            logger.debug(f"Cache hit: {cache_key}")
            text = path.read_text(encoding="utf-8")
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return text
        return None

    def _write_cache(self, cache_key: str, data: dict | str):
        """Write response data to cache."""
        path = self._cache_path(cache_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(data, str):
            path.write_text(data, encoding="utf-8")
        else:
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.debug(f"Cached: {cache_key}")

    async def fetch(
        self,
        url: str,
        cache_key: str | None = None,
        response_type: str = "json",
    ) -> dict | str | None:
        """Fetch a URL with caching, rate limiting, and retries.

        Args:
            url: The URL to fetch.
            cache_key: Key for local file cache. If None, derived from URL.
            response_type: "json" or "text".

        Returns:
            Parsed JSON dict or raw text, or None on failure.
        """
        if cache_key is None:
            cache_key = hashlib.md5(url.encode()).hexdigest()

        cached = self._read_cache(cache_key)
        if cached is not None:
            return cached

        await self._rate_limit()

        for attempt in range(1, self.max_retries + 1):
            try:
                async with httpx.AsyncClient(
                    headers=self.headers, timeout=30.0, follow_redirects=True
                ) as client:
                    resp = await client.get(url)

                if resp.status_code == 200:
                    if response_type == "json":
                        data = resp.json()
                    else:
                        data = resp.text
                    self._write_cache(cache_key, data)
                    logger.info(f"Fetched: {url} ({resp.status_code})")
                    return data

                if resp.status_code in (429, 500, 502, 503, 504):
                    wait = 2**attempt
                    logger.warning(
                        f"Retry {attempt}/{self.max_retries} for {url} "
                        f"(status {resp.status_code}), waiting {wait}s"
                    )
                    await asyncio.sleep(wait)
                    continue

                logger.error(f"Failed: {url} (status {resp.status_code})")
                return None

            except (httpx.RequestError, httpx.TimeoutException) as e:
                wait = 2**attempt
                logger.warning(
                    f"Retry {attempt}/{self.max_retries} for {url}: {e}, waiting {wait}s"
                )
                await asyncio.sleep(wait)

        logger.error(f"All retries exhausted for {url}")
        return None
