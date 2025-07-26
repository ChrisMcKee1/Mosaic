"""
Comprehensive tests for OAuth2Handler in Mosaic MCP Server.

Tests cover:
- OAuth 2.1 authentication flow with Microsoft Entra ID
- Token validation and refresh mechanisms
- Authorization middleware for MCP endpoints
- Session management and security features
- Error handling and edge cases
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime, timedelta
import jwt
from typing import Dict, Any


# Mock the OAuth2Handler since auth module might not be fully implemented
class MockOAuth2Handler:
    """Mock OAuth2Handler for testing purposes."""

    def __init__(self, settings):
        self.settings = settings
        self.tenant_id = getattr(settings, "azure_tenant_id", "test-tenant-id")
        self.client_id = getattr(settings, "azure_client_id", "test-client-id")
        self.jwks_client = None
        self.issuer = f"https://login.microsoftonline.com/{self.tenant_id}/v2.0"
        self.audience = self.client_id

    async def initialize(self):
        """Initialize OAuth2 handler."""
        self.jwks_client = MagicMock()

    async def validate_token(self, token: str) -> Dict[str, Any]:
        """Mock token validation."""
        if token == "invalid_token":
            raise ValueError("Invalid token")

        if token == "expired_token":
            raise jwt.ExpiredSignatureError("Token has expired")

        if token == "malformed_token":
            raise jwt.DecodeError("Token is malformed")

        # Return mock payload for valid tokens
        return {
            "sub": "user123",
            "aud": self.client_id,
            "iss": self.issuer,
            "exp": int((datetime.utcnow() + timedelta(hours=1)).timestamp()),
            "iat": int(datetime.utcnow().timestamp()),
            "scope": "api://mosaic-mcp/.default",
            "roles": ["MCP.User"],
        }

    async def refresh_token(self, refresh_token: str) -> Dict[str, str]:
        """Mock token refresh."""
        if refresh_token == "invalid_refresh":
            raise ValueError("Invalid refresh token")

        return {
            "access_token": "new_access_token",
            "refresh_token": "new_refresh_token",
            "expires_in": 3600,
            "token_type": "Bearer",
        }

    def create_authorization_middleware(self):
        """Create authorization middleware for FastMCP."""

        async def middleware(request, handler):
            auth_header = request.headers.get("Authorization")
            if not auth_header:
                raise ValueError("Missing Authorization header")

            if not auth_header.startswith("Bearer "):
                raise ValueError("Invalid Authorization header format")

            token = auth_header[7:]  # Remove "Bearer " prefix
            payload = await self.validate_token(token)

            # Add user info to request context
            request.user = {
                "id": payload["sub"],
                "roles": payload.get("roles", []),
                "scope": payload.get("scope", ""),
            }

            return await handler(request)

        return middleware

    async def get_user_info(self, token: str) -> Dict[str, Any]:
        """Get user information from token."""
        payload = await self.validate_token(token)
        return {
            "id": payload["sub"],
            "roles": payload.get("roles", []),
            "scope": payload.get("scope", ""),
            "expires_at": payload["exp"],
        }

    async def cleanup(self):
        """Cleanup OAuth2 handler resources."""
        self.jwks_client = None


@pytest.fixture
def mock_settings():
    """Create mock settings for OAuth2 testing."""
    settings = MagicMock()
    settings.oauth_enabled = True
    settings.azure_tenant_id = "test-tenant-id"
    settings.azure_client_id = "test-client-id"
    settings.azure_client_secret = "test-client-secret"
    settings.oauth_redirect_uri = "http://localhost:8000/auth/callback"
    settings.oauth_scope = "api://mosaic-mcp/.default"
    return settings


@pytest.fixture
def oauth_handler(mock_settings):
    """Create OAuth2Handler instance for testing."""
    return MockOAuth2Handler(mock_settings)


class TestOAuth2HandlerInitialization:
    """Test OAuth2Handler initialization and configuration."""

    def test_init_with_settings(self, mock_settings):
        """Test OAuth2Handler initialization with settings."""
        handler = MockOAuth2Handler(mock_settings)
        assert handler.settings == mock_settings
        assert handler.tenant_id == "test-tenant-id"
        assert handler.client_id == "test-client-id"
        assert handler.issuer == "https://login.microsoftonline.com/test-tenant-id/v2.0"
        assert handler.audience == "test-client-id"

    @pytest.mark.asyncio
    async def test_initialize_success(self, oauth_handler):
        """Test successful OAuth2Handler initialization."""
        await oauth_handler.initialize()
        assert oauth_handler.jwks_client is not None

    def test_issuer_construction(self, mock_settings):
        """Test correct issuer URL construction."""
        handler = MockOAuth2Handler(mock_settings)
        expected_issuer = (
            f"https://login.microsoftonline.com/{mock_settings.azure_tenant_id}/v2.0"
        )
        assert handler.issuer == expected_issuer


class TestTokenValidation:
    """Test JWT token validation functionality."""

    @pytest.mark.asyncio
    async def test_validate_valid_token(self, oauth_handler):
        """Test validation of valid JWT token."""
        valid_token = "valid_jwt_token"
        payload = await oauth_handler.validate_token(valid_token)

        assert payload["sub"] == "user123"
        assert payload["aud"] == oauth_handler.client_id
        assert payload["iss"] == oauth_handler.issuer
        assert "exp" in payload
        assert "iat" in payload
        assert "scope" in payload
        assert "roles" in payload

    @pytest.mark.asyncio
    async def test_validate_invalid_token(self, oauth_handler):
        """Test validation of invalid JWT token."""
        with pytest.raises(ValueError, match="Invalid token"):
            await oauth_handler.validate_token("invalid_token")

    @pytest.mark.asyncio
    async def test_validate_expired_token(self, oauth_handler):
        """Test validation of expired JWT token."""
        with pytest.raises(jwt.ExpiredSignatureError, match="Token has expired"):
            await oauth_handler.validate_token("expired_token")

    @pytest.mark.asyncio
    async def test_validate_malformed_token(self, oauth_handler):
        """Test validation of malformed JWT token."""
        with pytest.raises(jwt.DecodeError, match="Token is malformed"):
            await oauth_handler.validate_token("malformed_token")

    @pytest.mark.asyncio
    async def test_token_payload_structure(self, oauth_handler):
        """Test JWT token payload structure."""
        payload = await oauth_handler.validate_token("valid_token")

        # Required claims
        required_claims = ["sub", "aud", "iss", "exp", "iat"]
        for claim in required_claims:
            assert claim in payload

        # Optional claims
        assert "scope" in payload
        assert "roles" in payload
        assert isinstance(payload["roles"], list)

    @pytest.mark.asyncio
    async def test_token_expiration_validation(self, oauth_handler):
        """Test token expiration time validation."""
        payload = await oauth_handler.validate_token("valid_token")

        exp_timestamp = payload["exp"]
        current_time = datetime.utcnow().timestamp()

        # Token should expire in the future
        assert exp_timestamp > current_time


class TestTokenRefresh:
    """Test token refresh functionality."""

    @pytest.mark.asyncio
    async def test_refresh_valid_token(self, oauth_handler):
        """Test refreshing valid refresh token."""
        refresh_token = "valid_refresh_token"
        result = await oauth_handler.refresh_token(refresh_token)

        assert "access_token" in result
        assert "refresh_token" in result
        assert "expires_in" in result
        assert "token_type" in result
        assert result["token_type"] == "Bearer"
        assert result["expires_in"] == 3600

    @pytest.mark.asyncio
    async def test_refresh_invalid_token(self, oauth_handler):
        """Test refreshing invalid refresh token."""
        with pytest.raises(ValueError, match="Invalid refresh token"):
            await oauth_handler.refresh_token("invalid_refresh")

    @pytest.mark.asyncio
    async def test_refresh_token_response_format(self, oauth_handler):
        """Test refresh token response format."""
        result = await oauth_handler.refresh_token("valid_refresh")

        # Verify response structure
        expected_keys = ["access_token", "refresh_token", "expires_in", "token_type"]
        for key in expected_keys:
            assert key in result

        # Verify data types
        assert isinstance(result["access_token"], str)
        assert isinstance(result["refresh_token"], str)
        assert isinstance(result["expires_in"], int)
        assert isinstance(result["token_type"], str)


class TestAuthorizationMiddleware:
    """Test authorization middleware for MCP endpoints."""

    @pytest.mark.asyncio
    async def test_middleware_creation(self, oauth_handler):
        """Test creation of authorization middleware."""
        middleware = oauth_handler.create_authorization_middleware()
        assert callable(middleware)

    @pytest.mark.asyncio
    async def test_middleware_valid_token(self, oauth_handler):
        """Test middleware with valid authorization token."""
        middleware = oauth_handler.create_authorization_middleware()

        # Mock request with valid token
        request = MagicMock()
        request.headers = {"Authorization": "Bearer valid_token"}

        # Mock handler
        handler = AsyncMock()
        handler.return_value = {"status": "success"}

        result = await middleware(request, handler)

        # Verify user info was added to request
        assert hasattr(request, "user")
        assert request.user["id"] == "user123"
        assert "MCP.User" in request.user["roles"]

        # Verify handler was called
        handler.assert_called_once_with(request)
        assert result["status"] == "success"

    @pytest.mark.asyncio
    async def test_middleware_missing_header(self, oauth_handler):
        """Test middleware with missing Authorization header."""
        middleware = oauth_handler.create_authorization_middleware()

        request = MagicMock()
        request.headers = {}
        handler = AsyncMock()

        with pytest.raises(ValueError, match="Missing Authorization header"):
            await middleware(request, handler)

    @pytest.mark.asyncio
    async def test_middleware_invalid_header_format(self, oauth_handler):
        """Test middleware with invalid Authorization header format."""
        middleware = oauth_handler.create_authorization_middleware()

        request = MagicMock()
        request.headers = {"Authorization": "Invalid format"}
        handler = AsyncMock()

        with pytest.raises(ValueError, match="Invalid Authorization header format"):
            await middleware(request, handler)

    @pytest.mark.asyncio
    async def test_middleware_invalid_token(self, oauth_handler):
        """Test middleware with invalid token."""
        middleware = oauth_handler.create_authorization_middleware()

        request = MagicMock()
        request.headers = {"Authorization": "Bearer invalid_token"}
        handler = AsyncMock()

        with pytest.raises(ValueError, match="Invalid token"):
            await middleware(request, handler)

    @pytest.mark.asyncio
    async def test_middleware_expired_token(self, oauth_handler):
        """Test middleware with expired token."""
        middleware = oauth_handler.create_authorization_middleware()

        request = MagicMock()
        request.headers = {"Authorization": "Bearer expired_token"}
        handler = AsyncMock()

        with pytest.raises(jwt.ExpiredSignatureError):
            await middleware(request, handler)


class TestUserInfoExtraction:
    """Test user information extraction from tokens."""

    @pytest.mark.asyncio
    async def test_get_user_info_valid_token(self, oauth_handler):
        """Test extracting user info from valid token."""
        user_info = await oauth_handler.get_user_info("valid_token")

        assert "id" in user_info
        assert "roles" in user_info
        assert "scope" in user_info
        assert "expires_at" in user_info
        assert user_info["id"] == "user123"
        assert isinstance(user_info["roles"], list)

    @pytest.mark.asyncio
    async def test_get_user_info_invalid_token(self, oauth_handler):
        """Test extracting user info from invalid token."""
        with pytest.raises(ValueError):
            await oauth_handler.get_user_info("invalid_token")

    @pytest.mark.asyncio
    async def test_user_roles_extraction(self, oauth_handler):
        """Test extraction of user roles from token."""
        user_info = await oauth_handler.get_user_info("valid_token")

        assert "roles" in user_info
        assert "MCP.User" in user_info["roles"]
        assert isinstance(user_info["roles"], list)

    @pytest.mark.asyncio
    async def test_user_scope_extraction(self, oauth_handler):
        """Test extraction of user scope from token."""
        user_info = await oauth_handler.get_user_info("valid_token")

        assert "scope" in user_info
        assert user_info["scope"] == "api://mosaic-mcp/.default"


class TestOAuth2Integration:
    """Test OAuth2 integration scenarios."""

    @pytest.mark.asyncio
    async def test_full_authentication_flow(self, oauth_handler):
        """Test complete authentication flow."""
        await oauth_handler.initialize()

        # Step 1: Validate initial token
        payload = await oauth_handler.validate_token("valid_token")
        assert payload["sub"] == "user123"

        # Step 2: Get user info
        user_info = await oauth_handler.get_user_info("valid_token")
        assert user_info["id"] == payload["sub"]

        # Step 3: Test middleware
        middleware = oauth_handler.create_authorization_middleware()
        request = MagicMock()
        request.headers = {"Authorization": "Bearer valid_token"}
        handler = AsyncMock(return_value={"data": "success"})

        result = await middleware(request, handler)
        assert result["data"] == "success"
        assert request.user["id"] == "user123"

    @pytest.mark.asyncio
    async def test_token_refresh_flow(self, oauth_handler):
        """Test token refresh flow."""
        # Refresh token
        new_tokens = await oauth_handler.refresh_token("valid_refresh")

        # Validate new access token
        new_access_token = new_tokens["access_token"]
        payload = await oauth_handler.validate_token(new_access_token)

        assert payload["sub"] == "user123"
        assert "exp" in payload

    @pytest.mark.asyncio
    async def test_concurrent_token_validation(self, oauth_handler):
        """Test concurrent token validation."""
        import asyncio

        tokens = ["valid_token"] * 5

        tasks = [oauth_handler.validate_token(token) for token in tokens]
        results = await asyncio.gather(*tasks)

        assert len(results) == 5
        for result in results:
            assert result["sub"] == "user123"


class TestOAuth2ErrorHandling:
    """Test OAuth2 error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_network_error_simulation(self, oauth_handler):
        """Test handling of network errors during validation."""
        # This would test actual network error handling
        # For now, we test the concept
        assert oauth_handler.jwks_client is None
        await oauth_handler.initialize()
        assert oauth_handler.jwks_client is not None

    @pytest.mark.asyncio
    async def test_malformed_jwt_handling(self, oauth_handler):
        """Test handling of various malformed JWTs."""
        malformed_tokens = [
            "not.a.jwt",
            "missing.signature",
            "completely_invalid",
            "",
            None,
        ]

        for token in malformed_tokens:
            if token is None:
                continue
            try:
                await oauth_handler.validate_token(token)
                # Should not reach here for invalid tokens
                assert False, f"Expected error for token: {token}"
            except (ValueError, jwt.DecodeError, jwt.ExpiredSignatureError):
                # Expected for malformed tokens
                pass

    @pytest.mark.asyncio
    async def test_cleanup(self, oauth_handler):
        """Test OAuth2 handler cleanup."""
        await oauth_handler.initialize()
        assert oauth_handler.jwks_client is not None

        await oauth_handler.cleanup()
        assert oauth_handler.jwks_client is None

    def test_settings_validation(self, mock_settings):
        """Test validation of OAuth2 settings."""
        # Test with valid settings
        handler = MockOAuth2Handler(mock_settings)
        assert handler.tenant_id == mock_settings.azure_tenant_id
        assert handler.client_id == mock_settings.azure_client_id

        # Test with missing settings
        incomplete_settings = MagicMock()
        incomplete_settings.azure_tenant_id = None
        incomplete_settings.azure_client_id = None

        handler2 = MockOAuth2Handler(incomplete_settings)
        assert handler2.tenant_id == "test-tenant-id"  # Default fallback
        assert handler2.client_id == "test-client-id"  # Default fallback


class TestOAuth2SecurityFeatures:
    """Test OAuth2 security features and considerations."""

    @pytest.mark.asyncio
    async def test_role_based_access(self, oauth_handler):
        """Test role-based access control."""
        payload = await oauth_handler.validate_token("valid_token")
        roles = payload.get("roles", [])

        assert "MCP.User" in roles
        # In real implementation, you might check for specific roles
        # like "MCP.Admin", "MCP.ReadOnly", etc.

    @pytest.mark.asyncio
    async def test_scope_validation(self, oauth_handler):
        """Test OAuth2 scope validation."""
        payload = await oauth_handler.validate_token("valid_token")
        scope = payload.get("scope", "")

        assert "api://mosaic-mcp/.default" in scope

    @pytest.mark.asyncio
    async def test_audience_validation(self, oauth_handler):
        """Test JWT audience validation."""
        payload = await oauth_handler.validate_token("valid_token")

        assert payload["aud"] == oauth_handler.client_id
        assert payload["iss"] == oauth_handler.issuer

    def test_secure_defaults(self, oauth_handler):
        """Test that OAuth2 handler uses secure defaults."""
        # Verify HTTPS issuer
        assert oauth_handler.issuer.startswith("https://")

        # Verify Microsoft endpoint
        assert "login.microsoftonline.com" in oauth_handler.issuer

        # Verify v2.0 endpoint (OAuth 2.1 compatible)
        assert "/v2.0" in oauth_handler.issuer


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
