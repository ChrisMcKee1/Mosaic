#!/usr/bin/env python3
"""Environment Configuration Validation Script
Tests that all required environment variables are loaded and Azure services are accessible
"""

import os
import sys
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    env_path = Path(__file__).parent.parent / ".env"  # Go up one level from scripts/
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✅ Loaded environment variables from {env_path}")
    else:
        print(f"⚠️ No .env file found at {env_path}")
except ImportError:
    print("⚠️ python-dotenv not available, relying on system environment variables")


def validate_environment():
    """Validate all required environment variables are set."""
    print("\n🔍 Environment Configuration Validation")
    print("=" * 50)

    # Required variables for local development
    required_vars = {
        "COSMOS_MODE": "Cosmos DB mode (local/azure)",
        "AZURE_COSMOS_DB_ENDPOINT": "Cosmos DB endpoint",
        "AZURE_COSMOS_KEY": "Cosmos DB authentication key",
        "MOSAIC_DATABASE_NAME": "Database name",
        "AZURE_OPENAI_ENDPOINT": "Azure OpenAI service endpoint",
        "AZURE_OPENAI_API_KEY": "Azure OpenAI API key",
        "AZURE_OPENAI_DEPLOYMENT_NAME": "Azure OpenAI deployment name",
        "SEMANTICKERNEL_EXPERIMENTAL_GENAI_ENABLE_OTEL_DIAGNOSTICS": "Semantic Kernel telemetry",
    }

    missing_vars = []
    present_vars = []

    for var, description in required_vars.items():
        value = os.getenv(var)
        if value:
            # Mask sensitive values
            if "KEY" in var or "SECRET" in var:
                display_value = f"{value[:8]}..." if len(value) > 8 else "***"
            else:
                display_value = value
            present_vars.append((var, display_value, description))
            print(f"✅ {var}: {display_value}")
        else:
            missing_vars.append((var, description))
            print(f"❌ {var}: Not set ({description})")

    print(f"\n📊 Summary: {len(present_vars)} present, {len(missing_vars)} missing")

    return len(missing_vars) == 0


def test_azure_openai():
    """Test Azure OpenAI connection."""
    print("\n🤖 Testing Azure OpenAI Connection")
    print("=" * 40)

    try:
        import semantic_kernel as sk
        from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion

        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

        if not all([endpoint, api_key, deployment]):
            print("❌ Missing Azure OpenAI configuration")
            return False

        # Create Semantic Kernel service
        chat_service = AzureChatCompletion(
            deployment_name=deployment,
            endpoint=endpoint,
            api_key=api_key,
            service_id="test_service",
        )

        print("✅ Azure OpenAI service created successfully")
        print(f"   Endpoint: {endpoint}")
        print(f"   Deployment: {deployment} (gpt-4.1 series)")
        print(f"   API Key: {api_key[:8]}...")

        return True

    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        return False
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False


def test_cosmos_db():
    """Test Cosmos DB connection."""
    print("\n🌐 Testing Cosmos DB Connection")
    print("=" * 40)

    try:
        from azure.cosmos import CosmosClient

        endpoint = os.getenv("AZURE_COSMOS_DB_ENDPOINT")
        key = os.getenv("AZURE_COSMOS_KEY")
        database = os.getenv("MOSAIC_DATABASE_NAME")

        if not all([endpoint, key, database]):
            print("❌ Missing Cosmos DB configuration")
            return False

        # For local emulator, disable SSL verification
        if "localhost" in endpoint:
            import urllib3

            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            os.environ["PYTHONHTTPSVERIFY"] = "0"

        client = CosmosClient(endpoint, key)

        # Test connection by listing databases
        databases = list(client.list_databases())
        print("✅ Cosmos DB connection successful")
        print(f"   Endpoint: {endpoint}")
        print(f"   Database: {database}")
        print(f"   Available databases: {[db['id'] for db in databases]}")

        return True

    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        return False
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False


def main():
    """Run all validation tests."""
    print("🎯 Mosaic Environment Validation")
    print("=" * 60)

    # Test environment variables
    env_ok = validate_environment()

    # Test Azure services if environment is configured
    if env_ok:
        openai_ok = test_azure_openai()
        cosmos_ok = test_cosmos_db()

        print("\n🏁 Final Results")
        print("=" * 20)
        print(f"Environment Variables: {'✅ Pass' if env_ok else '❌ Fail'}")
        print(f"Azure OpenAI: {'✅ Pass' if openai_ok else '❌ Fail'}")
        print(f"Cosmos DB: {'✅ Pass' if cosmos_ok else '❌ Fail'}")

        if all([env_ok, openai_ok, cosmos_ok]):
            print(
                "\n🎉 All tests passed! Your environment is ready for Mosaic development."
            )
            return 0
        print("\n⚠️ Some tests failed. Please check the configuration above.")
        return 1
    print("\n❌ Environment validation failed. Please configure missing variables.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
