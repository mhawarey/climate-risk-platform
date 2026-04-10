"""
Cache utility for the Climate Risk Integration Platform.
Provides a simple caching system for API responses and processed data.
"""

import os
import json
import time
import logging
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime, timedelta


class Cache:
    """
    Simple file-based cache system for data
    """
    
    def __init__(self, cache_dir: str, max_age_hours: int = 24, enabled: bool = True):
        """
        Initialize cache
        
        Args:
            cache_dir: Directory to store cache files
            max_age_hours: Maximum age of cache entries in hours
            enabled: Whether caching is enabled
        """
        self.logger = logging.getLogger("cache")
        self.cache_dir = os.path.expanduser(cache_dir)
        self.max_age = max_age_hours * 3600  # Convert to seconds
        self.enabled = enabled
        
        # Cache directory will be created on first use, not at initialization
        self.logger.info(f"Cache initialized with directory: {self.cache_dir}")
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found or expired
        """
        if not self.enabled:
            return None
        
        # Generate file path from key
        cache_file = self._get_cache_file(key)
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                
                # Check if cache entry is expired
                timestamp = cache_data.get('timestamp', 0)
                if time.time() - timestamp <= self.max_age:
                    self.logger.debug(f"Cache hit for key: {key}")
                    return cache_data.get('data')
                else:
                    self.logger.debug(f"Cache expired for key: {key}")
                    # Delete expired cache file
                    os.unlink(cache_file)
            except Exception as e:
                self.logger.warning(f"Error reading cache for key {key}: {e}")
        
        return None
    
    def set(self, key: str, value: Any) -> bool:
        """
        Set value in cache
        
        Args:
            key: Cache key
            value: Value to cache
            
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            return False
        
        cache_file = self._get_cache_file(key)
        
        try:
            # Create cache data with timestamp
            cache_data = {
                'timestamp': time.time(),
                'data': value
            }
            
            # Create parent directories if they don't exist (on first use)
            self._ensure_cache_dir(cache_file)
            
            # Write to cache file
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f)
            
            self.logger.debug(f"Cache set for key: {key}")
            return True
        except Exception as e:
            self.logger.warning(f"Error writing cache for key {key}: {e}")
            return False
    
    def _ensure_cache_dir(self, cache_file: str) -> None:
        """
        Ensure cache directory exists - only created when needed
        
        Args:
            cache_file: Path to cache file
        """
        cache_dir = os.path.dirname(cache_file)
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
            self.logger.debug(f"Created cache directory: {cache_dir}")
    
    def delete(self, key: str) -> bool:
        """
        Delete value from cache
        
        Args:
            key: Cache key
            
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            return False
        
        cache_file = self._get_cache_file(key)
        
        if os.path.exists(cache_file):
            try:
                os.unlink(cache_file)
                self.logger.debug(f"Cache deleted for key: {key}")
                return True
            except Exception as e:
                self.logger.warning(f"Error deleting cache for key {key}: {e}")
        
        return False
    
    def clear(self, prefix: Optional[str] = None) -> int:
        """
        Clear all cache entries or those with a specific prefix
        
        Args:
            prefix: Optional prefix to filter cache keys
            
        Returns:
            Number of entries cleared
        """
        if not self.enabled or not os.path.exists(self.cache_dir):
            return 0
        
        count = 0
        
        try:
            for root, dirs, files in os.walk(self.cache_dir):
                for file in files:
                    if file.endswith('.cache'):
                        if prefix is None or self._key_matches_prefix(file, prefix):
                            os.unlink(os.path.join(root, file))
                            count += 1
            
            self.logger.info(f"Cleared {count} cache entries")
            return count
        except Exception as e:
            self.logger.warning(f"Error clearing cache: {e}")
            return 0
    
    def clear_expired(self) -> int:
        """
        Clear expired cache entries
        
        Returns:
            Number of expired entries cleared
        """
        if not self.enabled or not os.path.exists(self.cache_dir):
            return 0
        
        count = 0
        
        try:
            current_time = time.time()
            
            for root, dirs, files in os.walk(self.cache_dir):
                for file in files:
                    if file.endswith('.cache'):
                        cache_file = os.path.join(root, file)
                        
                        try:
                            with open(cache_file, 'r') as f:
                                cache_data = json.load(f)
                            
                            timestamp = cache_data.get('timestamp', 0)
                            if current_time - timestamp > self.max_age:
                                os.unlink(cache_file)
                                count += 1
                        except Exception:
                            # If we can't read the file, consider it corrupted and delete it
                            os.unlink(cache_file)
                            count += 1
            
            self.logger.info(f"Cleared {count} expired cache entries")
            return count
        except Exception as e:
            self.logger.warning(f"Error clearing expired cache: {e}")
            return 0
    
    def _get_cache_file(self, key: str) -> str:
        """
        Get cache file path for a key
        
        Args:
            key: Cache key
            
        Returns:
            Path to cache file
        """
        # Hash the key to create a filename
        key_hash = hashlib.md5(key.encode('utf-8')).hexdigest()
        
        # Use first two characters for subdirectory to avoid too many files in one directory
        subdir = key_hash[:2]
        
        return os.path.join(self.cache_dir, subdir, f"{key_hash}.cache")
    
    def _key_matches_prefix(self, cache_file: str, prefix: str) -> bool:
        """
        Check if a cache file corresponds to a key with the given prefix
        
        Args:
            cache_file: Cache file name
            prefix: Prefix to check
            
        Returns:
            True if the key has the prefix, False otherwise
        """
        # This is a simplified check - in a real implementation, we might
        # store the original key in the cache file metadata
        prefix_hash = hashlib.md5(prefix.encode('utf-8')).hexdigest()
        return cache_file.startswith(prefix_hash[:8])
    
    def flush(self) -> bool:
        """
        Ensure all pending cache writes are flushed to disk
        
        Returns:
            True if successful, False otherwise
        """
        # Since we're using file-based caching, no explicit flush is needed
        # File writes are committed when the file is closed
        return True