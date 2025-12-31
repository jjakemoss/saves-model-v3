"""SQLite-based cache manager for NHL API responses"""

import sqlite3
import json
import hashlib
from typing import Optional, Any
from datetime import datetime, timedelta
import logging


class CacheManager:
    """
    SQLite-based cache for NHL API responses
    - TTL-based expiration (24 hours default)
    - Cache invalidation methods
    - Reduces API load for repeated requests
    """

    def __init__(self, db_path: str = "data/cache/api_cache.db", ttl_seconds: int = 86400):
        """
        Initialize cache manager

        Args:
            db_path: Path to SQLite database
            ttl_seconds: Time-to-live for cached entries (default 24 hours)
        """
        self.db_path = db_path
        self.ttl_seconds = ttl_seconds
        self.logger = logging.getLogger(__name__)

        # Create cache table
        self._init_db()

    def _init_db(self):
        """Initialize SQLite database with cache table"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS api_cache (
                    cache_key TEXT PRIMARY KEY,
                    response_data TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    expires_at TIMESTAMP NOT NULL
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_expires_at
                ON api_cache(expires_at)
            """)
            conn.commit()
            self.logger.info(f"Cache database initialized at {self.db_path}")

    def _generate_cache_key(self, endpoint: str, params: Optional[dict] = None) -> str:
        """
        Generate unique cache key from endpoint and parameters

        Args:
            endpoint: API endpoint
            params: Query parameters

        Returns:
            MD5 hash of endpoint + params
        """
        key_str = endpoint
        if params:
            # Sort params for consistent hashing
            sorted_params = sorted(params.items())
            key_str += str(sorted_params)

        return hashlib.md5(key_str.encode()).hexdigest()

    def get(self, endpoint: str, params: Optional[dict] = None) -> Optional[dict]:
        """
        Retrieve cached response if available and not expired

        Args:
            endpoint: API endpoint
            params: Query parameters

        Returns:
            Cached response dict or None if not found/expired
        """
        cache_key = self._generate_cache_key(endpoint, params)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT response_data, expires_at
                FROM api_cache
                WHERE cache_key = ?
            """, (cache_key,))

            result = cursor.fetchone()

            if result:
                response_data, expires_at = result
                expires_at = datetime.fromisoformat(expires_at)

                # Check if expired
                if datetime.now() < expires_at:
                    self.logger.debug(f"Cache HIT: {endpoint}")
                    return json.loads(response_data)
                else:
                    # Delete expired entry
                    self.logger.debug(f"Cache EXPIRED: {endpoint}")
                    self.delete(cache_key)

            self.logger.debug(f"Cache MISS: {endpoint}")
            return None

    def set(self, endpoint: str, params: Optional[dict], response_data: dict, ttl_override: Optional[int] = None):
        """
        Store response in cache

        Args:
            endpoint: API endpoint
            params: Query parameters
            response_data: Response to cache
            ttl_override: Override default TTL (seconds)
        """
        cache_key = self._generate_cache_key(endpoint, params)
        created_at = datetime.now()
        ttl = ttl_override if ttl_override is not None else self.ttl_seconds
        expires_at = created_at + timedelta(seconds=ttl)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO api_cache
                (cache_key, response_data, created_at, expires_at)
                VALUES (?, ?, ?, ?)
            """, (
                cache_key,
                json.dumps(response_data),
                created_at.isoformat(),
                expires_at.isoformat()
            ))
            conn.commit()
            self.logger.debug(f"Cache SET: {endpoint} (expires in {ttl}s)")

    def delete(self, cache_key: str):
        """Delete specific cache entry"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM api_cache WHERE cache_key = ?", (cache_key,))
            conn.commit()

    def clear_expired(self):
        """Remove all expired cache entries"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM api_cache
                WHERE expires_at < ?
            """, (datetime.now().isoformat(),))
            deleted_count = cursor.rowcount
            conn.commit()
            self.logger.info(f"Cleared {deleted_count} expired cache entries")
            return deleted_count

    def clear_all(self):
        """Clear entire cache"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM api_cache")
            deleted_count = cursor.rowcount
            conn.commit()
            self.logger.info(f"Cleared all cache ({deleted_count} entries)")
            return deleted_count

    def get_stats(self) -> dict:
        """Get cache statistics"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Total entries
            cursor.execute("SELECT COUNT(*) FROM api_cache")
            total_entries = cursor.fetchone()[0]

            # Expired entries
            cursor.execute("""
                SELECT COUNT(*) FROM api_cache
                WHERE expires_at < ?
            """, (datetime.now().isoformat(),))
            expired_entries = cursor.fetchone()[0]

            # Active entries
            active_entries = total_entries - expired_entries

            return {
                "total_entries": total_entries,
                "active_entries": active_entries,
                "expired_entries": expired_entries,
                "db_path": self.db_path
            }


class CachedNHLAPIClient:
    """NHL API Client with caching layer"""

    def __init__(self, api_client, cache_manager: CacheManager):
        """
        Initialize cached API client

        Args:
            api_client: NHLAPIClient instance
            cache_manager: CacheManager instance
        """
        self.api = api_client
        self.cache = cache_manager
        self.logger = logging.getLogger(__name__)

    def _get_cached_or_fetch(self, endpoint: str, params: Optional[dict] = None, ttl_override: Optional[int] = None) -> dict:
        """
        Try to get from cache, otherwise fetch from API and cache

        Args:
            endpoint: API endpoint
            params: Query parameters
            ttl_override: Override default cache TTL

        Returns:
            Response data
        """
        # Try cache first
        cached_response = self.cache.get(endpoint, params)
        if cached_response is not None:
            return cached_response

        # Fetch from API
        response = self.api._make_request(endpoint, params)

        # Cache response
        self.cache.set(endpoint, params, response, ttl_override)

        return response

    # Wrap all API methods with caching
    def get_schedule_by_date(self, date: str) -> dict:
        return self._get_cached_or_fetch(f"/v1/schedule/{date}")

    def get_team_season_schedule(self, team: str, season: str) -> dict:
        # Long TTL for historical schedules (7 days)
        ttl = 604800 if season != "20242025" else 3600  # 1 hour for current season
        return self._get_cached_or_fetch(f"/v1/club-schedule-season/{team}/{season}", ttl_override=ttl)

    def get_boxscore(self, game_id: int) -> dict:
        # Long TTL for completed games (7 days)
        return self._get_cached_or_fetch(f"/v1/gamecenter/{game_id}/boxscore", ttl_override=604800)

    def get_play_by_play(self, game_id: int) -> dict:
        # Long TTL for completed games
        return self._get_cached_or_fetch(f"/v1/gamecenter/{game_id}/play-by-play", ttl_override=604800)

    def get_player_game_log(self, player_id: int, season: str, game_type: int = 2) -> dict:
        endpoint = f"/v1/player/{player_id}/game-log/{season}/{game_type}"
        # Current season: 1 hour TTL, historical: 7 days
        ttl = 604800 if season != "20242025" else 3600
        return self._get_cached_or_fetch(endpoint, ttl_override=ttl)

    def get_club_stats(self, team: str, season: str, game_type: int = 2) -> dict:
        endpoint = f"/v1/club-stats/{team}/{season}/{game_type}"
        ttl = 604800 if season != "20242025" else 3600
        return self._get_cached_or_fetch(endpoint, ttl_override=ttl)

    def get_goalie_shot_location_detail(self, player_id: int, season: str, game_type: int = 2) -> dict:
        endpoint = f"/v1/edge/goalie-shot-location-detail/{player_id}/{season}/{game_type}"
        ttl = 604800 if season != "20242025" else 3600
        return self._get_cached_or_fetch(endpoint, ttl_override=ttl)

    def get_goalie_comparison(self, player_id: int, season: str, game_type: int = 2) -> dict:
        endpoint = f"/v1/edge/goalie-comparison/{player_id}/{season}/{game_type}"
        ttl = 604800 if season != "20242025" else 3600
        return self._get_cached_or_fetch(endpoint, ttl_override=ttl)

    def get_goalie_5v5_detail(self, player_id: int, season: str, game_type: int = 2) -> dict:
        endpoint = f"/v1/edge/goalie-5v5-detail/{player_id}/{season}/{game_type}"
        ttl = 604800 if season != "20242025" else 3600
        return self._get_cached_or_fetch(endpoint, ttl_override=ttl)

    def get_team_comparison(self, team_id: int, season: str, game_type: int = 2) -> dict:
        endpoint = f"/v1/edge/team-comparison/{team_id}/{season}/{game_type}"
        ttl = 604800 if season != "20242025" else 3600
        return self._get_cached_or_fetch(endpoint, ttl_override=ttl)
