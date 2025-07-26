"""
Azure Redis session management for query learning system.

This module provides Redis-based session persistence for the OMR-P3-004
query learning and adaptation system.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Union
from uuid import UUID

import redis.asyncio as aioredis
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from pydantic import ValidationError

from ..models.session_models import QuerySession, SessionSummary, ModelState
from ..config.settings import get_settings


logger = logging.getLogger(__name__)


class RedisSessionError(Exception):
    """Redis session operation errors."""

    pass


class SessionManager:
    """
    Azure Redis-based session management for query learning.

    Provides async operations for storing and retrieving query sessions,
    with support for Azure authentication and automatic serialization.
    """

    def __init__(
        self,
        redis_url: Optional[str] = None,
        key_vault_url: Optional[str] = None,
        session_ttl_hours: int = 24,
        max_interactions_per_session: int = 1000,
    ):
        """
        Initialize session manager.

        Args:
            redis_url: Redis connection URL (if None, uses environment)
            key_vault_url: Azure Key Vault URL for credentials
            session_ttl_hours: Session expiration time in hours
            max_interactions_per_session: Maximum interactions per session
        """
        self.settings = get_settings()
        self.redis_url = redis_url or self._get_redis_url()
        self.key_vault_url = key_vault_url
        self.session_ttl = timedelta(hours=session_ttl_hours)
        self.max_interactions = max_interactions_per_session

        self._redis: Optional[aioredis.Redis] = None
        self._connection_lock = asyncio.Lock()

        # Key prefixes for organization
        self.SESSION_PREFIX = "mosaic:session:"
        self.USER_SESSIONS_PREFIX = "mosaic:user_sessions:"
        self.SESSION_INDEX_PREFIX = "mosaic:session_index"
        self.MODEL_STATE_PREFIX = "mosaic:model_state:"

    def _get_redis_url(self) -> str:
        """Get Redis URL from settings or environment."""
        if hasattr(self.settings, "redis_url") and self.settings.redis_url:
            return self.settings.redis_url

        # Construct from Azure environment variables
        redis_host = self.settings.azure_redis_cache_name
        if redis_host and not redis_host.endswith(".redis.cache.windows.net"):
            redis_host = f"{redis_host}.redis.cache.windows.net"

        return f"rediss://{redis_host}:6380"

    async def _get_redis_password(self) -> str:
        """Get Redis password from Key Vault or environment."""
        if self.key_vault_url:
            try:
                credential = DefaultAzureCredential()
                client = SecretClient(
                    vault_url=self.key_vault_url, credential=credential
                )
                secret = client.get_secret("redis-primary-key")
                return secret.value
            except Exception as e:
                logger.warning(f"Failed to get Redis password from Key Vault: {e}")

        # Fallback to environment variable
        return self.settings.azure_redis_primary_key or ""

    async def _connect(self) -> aioredis.Redis:
        """Establish Redis connection with Azure authentication."""
        if self._redis is not None:
            try:
                await self._redis.ping()
                return self._redis
            except Exception:
                logger.info("Redis connection lost, reconnecting...")
                self._redis = None

        async with self._connection_lock:
            if self._redis is not None:
                return self._redis

            try:
                password = await self._get_redis_password()

                self._redis = aioredis.from_url(
                    self.redis_url,
                    password=password,
                    ssl_cert_reqs=None,
                    socket_connect_timeout=5,
                    socket_timeout=5,
                    retry_on_timeout=True,
                    health_check_interval=30,
                )

                # Test connection
                await self._redis.ping()
                logger.info("Successfully connected to Azure Redis Cache")

                return self._redis

            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                raise RedisSessionError(f"Redis connection failed: {e}")

    async def close(self) -> None:
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()
            self._redis = None

    def _session_key(self, session_id: Union[str, UUID]) -> str:
        """Generate Redis key for session."""
        return f"{self.SESSION_PREFIX}{session_id}"

    def _user_sessions_key(self, user_id: str) -> str:
        """Generate Redis key for user session list."""
        return f"{self.USER_SESSIONS_PREFIX}{user_id}"

    def _model_state_key(self, session_id: Union[str, UUID], model_name: str) -> str:
        """Generate Redis key for model state."""
        return f"{self.MODEL_STATE_PREFIX}{session_id}:{model_name}"

    async def create_session(
        self, user_id: Optional[str] = None, session_id: Optional[UUID] = None
    ) -> QuerySession:
        """
        Create a new query session.

        Args:
            user_id: Optional user identifier
            session_id: Optional session ID (generates if None)

        Returns:
            New QuerySession instance
        """
        session = QuerySession(
            session_id=session_id or UUID(),
            user_id=user_id,
            expires_at=datetime.now(timezone.utc) + self.session_ttl,
        )

        await self.save_session(session)

        # Add to user session index if user_id provided
        if user_id:
            await self._add_to_user_sessions(user_id, session.session_id)

        logger.info(f"Created session {session.session_id} for user {user_id}")
        return session

    async def get_session(self, session_id: Union[str, UUID]) -> Optional[QuerySession]:
        """
        Retrieve session by ID.

        Args:
            session_id: Session identifier

        Returns:
            QuerySession if found, None otherwise
        """
        try:
            redis = await self._connect()
            key = self._session_key(session_id)

            data = await redis.get(key)
            if not data:
                return None

            session_dict = json.loads(data)
            session = QuerySession.model_validate(session_dict)

            # Check expiration
            if session.expires_at and datetime.now(timezone.utc) > session.expires_at:
                await self.delete_session(session_id)
                return None

            return session

        except (json.JSONDecodeError, ValidationError) as e:
            logger.error(f"Failed to deserialize session {session_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error getting session {session_id}: {e}")
            return None

    async def save_session(self, session: QuerySession) -> bool:
        """
        Save session to Redis.

        Args:
            session: QuerySession to save

        Returns:
            True if successful, False otherwise
        """
        try:
            redis = await self._connect()
            key = self._session_key(session.session_id)

            # Limit interaction history to prevent memory issues
            if len(session.interactions) > self.max_interactions:
                session.interactions = session.interactions[-self.max_interactions :]

            # Serialize session
            session_data = session.model_dump(mode="json")
            serialized = json.dumps(session_data, default=str)

            # Calculate TTL
            ttl_seconds = None
            if session.expires_at:
                ttl_delta = session.expires_at - datetime.now(timezone.utc)
                ttl_seconds = max(int(ttl_delta.total_seconds()), 60)

            # Save with TTL
            if ttl_seconds:
                await redis.setex(key, ttl_seconds, serialized)
            else:
                await redis.set(key, serialized)

            logger.debug(f"Saved session {session.session_id}")
            return True

        except Exception as e:
            logger.error(f"Error saving session {session.session_id}: {e}")
            return False

    async def delete_session(self, session_id: Union[str, UUID]) -> bool:
        """
        Delete session and associated data.

        Args:
            session_id: Session identifier

        Returns:
            True if successful, False otherwise
        """
        try:
            redis = await self._connect()

            # Get session to find user_id
            session = await self.get_session(session_id)

            # Delete main session
            key = self._session_key(session_id)
            await redis.delete(key)

            # Remove from user session index
            if session and session.user_id:
                await self._remove_from_user_sessions(session.user_id, session_id)

            # Delete model states
            model_pattern = f"{self.MODEL_STATE_PREFIX}{session_id}:*"
            model_keys = []
            async for key in redis.scan_iter(match=model_pattern):
                model_keys.append(key)

            if model_keys:
                await redis.delete(*model_keys)

            logger.info(f"Deleted session {session_id}")
            return True

        except Exception as e:
            logger.error(f"Error deleting session {session_id}: {e}")
            return False

    async def get_user_sessions(self, user_id: str) -> List[SessionSummary]:
        """
        Get all sessions for a user.

        Args:
            user_id: User identifier

        Returns:
            List of session summaries
        """
        try:
            redis = await self._connect()
            key = self._user_sessions_key(user_id)

            session_ids = await redis.smembers(key)
            summaries = []

            for session_id_bytes in session_ids:
                session_id = session_id_bytes.decode("utf-8")
                session = await self.get_session(session_id)

                if session:
                    summary = SessionSummary(
                        session_id=session.session_id,
                        user_id=session.user_id,
                        status=session.status,
                        created_at=session.created_at,
                        last_activity=session.last_activity,
                        interaction_count=len(session.interactions),
                        feedback_ratio=session.get_feedback_ratio(),
                        current_strategy=session.current_strategy,
                    )
                    summaries.append(summary)
                else:
                    # Clean up dead reference
                    await redis.srem(key, session_id)

            return sorted(summaries, key=lambda x: x.last_activity, reverse=True)

        except Exception as e:
            logger.error(f"Error getting user sessions for {user_id}: {e}")
            return []

    async def save_model_state(
        self, session_id: Union[str, UUID], model_name: str, model_state: ModelState
    ) -> bool:
        """
        Save model state for a session.

        Args:
            session_id: Session identifier
            model_name: Name of the model
            model_state: Serialized model state

        Returns:
            True if successful, False otherwise
        """
        try:
            redis = await self._connect()
            key = self._model_state_key(session_id, model_name)

            state_data = model_state.model_dump(mode="json")
            serialized = json.dumps(state_data, default=str)

            # Set TTL to match session TTL
            ttl_seconds = int(self.session_ttl.total_seconds())
            await redis.setex(key, ttl_seconds, serialized)

            logger.debug(f"Saved model state {model_name} for session {session_id}")
            return True

        except Exception as e:
            logger.error(
                f"Error saving model state {model_name} for session {session_id}: {e}"
            )
            return False

    async def get_model_state(
        self, session_id: Union[str, UUID], model_name: str
    ) -> Optional[ModelState]:
        """
        Get model state for a session.

        Args:
            session_id: Session identifier
            model_name: Name of the model

        Returns:
            ModelState if found, None otherwise
        """
        try:
            redis = await self._connect()
            key = self._model_state_key(session_id, model_name)

            data = await redis.get(key)
            if not data:
                return None

            state_dict = json.loads(data)
            return ModelState.model_validate(state_dict)

        except (json.JSONDecodeError, ValidationError) as e:
            logger.error(
                f"Failed to deserialize model state {model_name} for session {session_id}: {e}"
            )
            return None
        except Exception as e:
            logger.error(
                f"Error getting model state {model_name} for session {session_id}: {e}"
            )
            return None

    async def _add_to_user_sessions(self, user_id: str, session_id: UUID) -> None:
        """Add session to user's session set."""
        try:
            redis = await self._connect()
            key = self._user_sessions_key(user_id)
            await redis.sadd(key, str(session_id))
            # Set TTL on user session index
            await redis.expire(key, int(self.session_ttl.total_seconds() * 2))
        except Exception as e:
            logger.error(f"Error adding session to user index: {e}")

    async def _remove_from_user_sessions(
        self, user_id: str, session_id: Union[str, UUID]
    ) -> None:
        """Remove session from user's session set."""
        try:
            redis = await self._connect()
            key = self._user_sessions_key(user_id)
            await redis.srem(key, str(session_id))
        except Exception as e:
            logger.error(f"Error removing session from user index: {e}")

    async def cleanup_expired_sessions(self) -> int:
        """
        Clean up expired sessions.

        Returns:
            Number of sessions cleaned up
        """
        try:
            redis = await self._connect()

            # Scan for all session keys
            session_keys = []
            pattern = f"{self.SESSION_PREFIX}*"
            async for key in redis.scan_iter(match=pattern):
                session_keys.append(key)

            cleaned_count = 0
            for key in session_keys:
                try:
                    ttl = await redis.ttl(key)
                    if ttl == -2:  # Key doesn't exist
                        cleaned_count += 1
                    elif ttl == -1:  # Key exists but no TTL
                        # Check if session is actually expired
                        data = await redis.get(key)
                        if data:
                            session_dict = json.loads(data)
                            expires_at_str = session_dict.get("expires_at")
                            if expires_at_str:
                                expires_at = datetime.fromisoformat(
                                    expires_at_str.replace("Z", "+00:00")
                                )
                                if datetime.now(timezone.utc) > expires_at:
                                    session_id = key.decode("utf-8").replace(
                                        self.SESSION_PREFIX, ""
                                    )
                                    await self.delete_session(session_id)
                                    cleaned_count += 1
                except Exception as e:
                    logger.warning(f"Error checking session key {key}: {e}")

            if cleaned_count > 0:
                logger.info(f"Cleaned up {cleaned_count} expired sessions")

            return cleaned_count

        except Exception as e:
            logger.error(f"Error cleaning up expired sessions: {e}")
            return 0
