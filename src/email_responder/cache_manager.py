"""Cache manager for prompt and embedding reuse."""

import json
import hashlib
import logging
from typing import Any, Optional, Dict, Union
from datetime import datetime, timedelta

try:
    import redis as redis_module
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis_module = None

from .config import config

logger = logging.getLogger(__name__)


class CacheManager:
    """Manages caching for prompts, embeddings, and responses."""
    
    def __init__(self):
        self.memory_cache: Dict[str, Any] = {}
        self.redis_client: Optional[Any] = None
        
        if config.use_redis_cache and REDIS_AVAILABLE and redis_module:
            try:
                self.redis_client = redis_module.from_url(config.redis_url)
                self.redis_client.ping()
                logger.info("Connected to Redis cache")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}. Using memory cache.")
                self.redis_client = None
        elif config.use_redis_cache and not REDIS_AVAILABLE:
            logger.warning("Redis not available. Using memory cache.")
    
    def _get_cache_key(self, prefix: str, data: str) -> str:
        """Generate a cache key from data."""
        hash_obj = hashlib.md5(data.encode('utf-8'))
        return f"{prefix}:{hash_obj.hexdigest()}"
    
    def _is_expired(self, timestamp: str) -> bool:
        """Check if cache entry is expired."""
        try:
            cache_time = datetime.fromisoformat(timestamp)
            expiry_time = cache_time + timedelta(hours=config.cache_ttl_hours)
            return datetime.now() > expiry_time
        except Exception:
            return True
    
    def get(self, prefix: str, key: str) -> Optional[Any]:
        """Get value from cache."""
        cache_key = self._get_cache_key(prefix, key)
        
        # Try Redis first
        if self.redis_client:
            try:
                cached_data = self.redis_client.get(cache_key)
                if cached_data:
                    # Handle bytes returned by Redis
                    if isinstance(cached_data, bytes):
                        cached_data = cached_data.decode('utf-8')
                    data = json.loads(cached_data)
                    if not self._is_expired(data['timestamp']):
                        logger.debug(f"Cache hit (Redis): {cache_key}")
                        return data['value']
                    else:
                        self.redis_client.delete(cache_key)
            except Exception as e:
                logger.warning(f"Redis cache error: {e}")
        
        # Fallback to memory cache
        if cache_key in self.memory_cache:
            data = self.memory_cache[cache_key]
            if not self._is_expired(data['timestamp']):
                logger.debug(f"Cache hit (memory): {cache_key}")
                return data['value']
            else:
                del self.memory_cache[cache_key]
        
        logger.debug(f"Cache miss: {cache_key}")
        return None
    
    def set(self, prefix: str, key: str, value: Any) -> None:
        """Set value in cache."""
        cache_key = self._get_cache_key(prefix, key)
        cache_data = {
            'value': value,
            'timestamp': datetime.now().isoformat()
        }
        
        # Store in Redis if available
        if self.redis_client:
            try:
                ttl_seconds = int(timedelta(hours=config.cache_ttl_hours).total_seconds())
                self.redis_client.setex(
                    cache_key,
                    ttl_seconds,
                    json.dumps(cache_data, default=str)
                )
                logger.debug(f"Cached in Redis: {cache_key}")
            except Exception as e:
                logger.warning(f"Failed to cache in Redis: {e}")
        
        # Always store in memory as backup
        self.memory_cache[cache_key] = cache_data
        logger.debug(f"Cached in memory: {cache_key}")
    
    def get_prompt_response(self, prompt: str) -> Optional[str]:
        """Get cached response for a prompt."""
        return self.get("prompt", prompt)
    
    def set_prompt_response(self, prompt: str, response: str) -> None:
        """Cache a prompt-response pair."""
        self.set("prompt", prompt, response)
    
    def get_embedding(self, text: str) -> Optional[list]:
        """Get cached embedding for text."""
        return self.get("embedding", text)
    
    def set_embedding(self, text: str, embedding: list) -> None:
        """Cache text embedding."""
        self.set("embedding", text, embedding)
    
    def clear_cache(self) -> None:
        """Clear all caches."""
        self.memory_cache.clear()
        if self.redis_client:
            try:
                self.redis_client.flushall()
                logger.info("Cleared Redis cache")
            except Exception as e:
                logger.warning(f"Failed to clear Redis cache: {e}")
        logger.info("Cleared memory cache")


# Global cache instance
cache_manager = CacheManager() 