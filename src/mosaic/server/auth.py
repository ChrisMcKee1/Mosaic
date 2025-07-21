"""
OAuth 2.1 Authentication Handler for Mosaic MCP Tool

Implements FR-14 requirement for secure MCP endpoint with OAuth 2.1 authentication
using Microsoft Entra ID as the identity provider.
"""

import logging
from typing import Optional, Dict, Any
import jwt
from datetime import datetime, timedelta

from azure.identity import DefaultAzureCredential
import httpx

from ..config.settings import MosaicSettings


logger = logging.getLogger(__name__)


class OAuth2Handler:
    """
    OAuth 2.1 authentication handler using Microsoft Entra ID.

    Provides secure authentication for MCP client connections following
    the MCP Authorization specification.
    """

    def __init__(self, settings: MosaicSettings):
        """Initialize OAuth2 handler."""
        self.settings = settings
        self.credential = DefaultAzureCredential()
        self.token_cache: Dict[str, Dict[str, Any]] = {}
        self.jwks_cache: Optional[Dict[str, Any]] = None
        self.jwks_cache_expiry: Optional[datetime] = None

    async def initialize(self) -> None:
        """Initialize OAuth2 handler and validate configuration."""
        if not self.settings.oauth_enabled:
            logger.info("OAuth 2.1 authentication disabled")
            return

        if not all([self.settings.azure_tenant_id, self.settings.azure_client_id]):
            raise ValueError(
                "OAuth 2.1 enabled but missing required configuration: "
                "azure_tenant_id and azure_client_id are required"
            )

        # Pre-load JWKS for token validation
        await self._load_jwks()

        logger.info("OAuth 2.1 authentication initialized")

    async def validate_token(self, token: str) -> Dict[str, Any]:
        """
        Validate an OAuth 2.1 access token.

        Args:
            token: JWT access token from client

        Returns:
            Dict containing token claims if valid

        Raises:
            ValueError: If token is invalid
        """
        if not self.settings.oauth_enabled:
            # If OAuth is disabled, return mock claims for development
            return {"sub": "development", "aud": "mosaic-mcp"}

        try:
            # Check cache first
            if token in self.token_cache:
                cached_data = self.token_cache[token]
                if cached_data["expires_at"] > datetime.utcnow():
                    return cached_data["claims"]
                else:
                    # Remove expired token from cache
                    del self.token_cache[token]

            # Decode and validate token
            claims = await self._decode_and_validate_token(token)

            # Cache valid token
            self.token_cache[token] = {
                "claims": claims,
                "expires_at": datetime.utcnow()
                + timedelta(minutes=55),  # Cache for 55 minutes
            }

            return claims

        except Exception as e:
            logger.error(f"Token validation failed: {e}")
            raise ValueError(f"Invalid token: {e}")

    async def _decode_and_validate_token(self, token: str) -> Dict[str, Any]:
        """Decode and validate JWT token using Azure AD JWKS."""
        try:
            # Decode header to get kid
            header = jwt.get_unverified_header(token)
            kid = header.get("kid")

            if not kid:
                raise ValueError("Token missing kid in header")

            # Get signing key from JWKS
            signing_key = await self._get_signing_key(kid)

            # Decode and validate token
            claims = jwt.decode(
                token,
                signing_key,
                algorithms=["RS256"],
                audience=self.settings.azure_client_id,
                issuer=f"https://sts.windows.net/{self.settings.azure_tenant_id}/",
                options={
                    "verify_signature": True,
                    "verify_exp": True,
                    "verify_aud": True,
                    "verify_iss": True,
                },
            )

            return claims

        except jwt.InvalidTokenError as e:
            raise ValueError(f"JWT validation failed: {e}")

    async def _get_signing_key(self, kid: str) -> str:
        """Get signing key from JWKS for the given key ID."""
        jwks = await self._get_jwks()

        for key in jwks.get("keys", []):
            if key.get("kid") == kid:
                # Convert JWK to PEM format for PyJWT
                return jwt.algorithms.RSAAlgorithm.from_jwk(key)

        raise ValueError(f"Signing key not found for kid: {kid}")

    async def _get_jwks(self) -> Dict[str, Any]:
        """Get JSON Web Key Set from Azure AD."""
        # Check cache first
        if (
            self.jwks_cache
            and self.jwks_cache_expiry
            and self.jwks_cache_expiry > datetime.utcnow()
        ):
            return self.jwks_cache

        return await self._load_jwks()

    async def _load_jwks(self) -> Dict[str, Any]:
        """Load JWKS from Azure AD well-known endpoint."""
        jwks_url = (
            f"https://login.microsoftonline.com/{self.settings.azure_tenant_id}"
            f"/discovery/v2.0/keys"
        )

        async with httpx.AsyncClient() as client:
            response = await client.get(jwks_url)
            response.raise_for_status()

            jwks = response.json()

            # Cache for 24 hours
            self.jwks_cache = jwks
            self.jwks_cache_expiry = datetime.utcnow() + timedelta(hours=24)

            return jwks

    def extract_token_from_header(
        self, authorization_header: Optional[str]
    ) -> Optional[str]:
        """Extract bearer token from Authorization header."""
        if not authorization_header:
            return None

        if not authorization_header.startswith("Bearer "):
            return None

        return authorization_header[7:]  # Remove "Bearer " prefix

    async def cleanup(self) -> None:
        """Cleanup authentication resources."""
        self.token_cache.clear()
        self.jwks_cache = None
        self.jwks_cache_expiry = None
        logger.info("OAuth2 handler cleanup completed")
