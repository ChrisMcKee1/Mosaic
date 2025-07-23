#!/usr/bin/env python3
"""
Ingestion Service Diagnostic Tests
Comprehensive diagnostics for ingestion service components and dependencies
"""

import sys
import asyncio
import traceback
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def test_imports():
    """Test basic imports to identify the failing component"""
    print("🧪 Testing basic imports...")

    try:
        print("   ✓ Testing Python standard library imports...")
        print("   ✅ Standard library imports successful")
    except Exception as e:
        print(f"   ❌ Standard library import failed: {e}")
        return False

    try:
        print("   ✓ Testing pydantic imports...")
        print("   ✅ Pydantic imports successful")
    except Exception as e:
        print(f"   ❌ Pydantic import failed: {e}")
        print("      Try: pip install pydantic pydantic-settings")
        return False

    try:
        print("   ✓ Testing mosaic settings...")
        print("   ✅ Mosaic settings import successful")
    except Exception as e:
        print(f"   ❌ Mosaic settings import failed: {e}")
        traceback.print_exc()
        return False

    try:
        print("   ✓ Testing GitPython...")
        print("   ✅ GitPython import successful")
    except Exception as e:
        print(f"   ❌ GitPython import failed: {e}")
        print("      Try: pip install GitPython")
        return False

    # Test optional heavy dependencies
    heavy_deps = {
        "tree-sitter": "tree_sitter",
        "semantic-kernel": "semantic_kernel",
        "azure-cosmos": "azure.cosmos",
        "azure-identity": "azure.identity",
    }

    for dep_name, module_name in heavy_deps.items():
        try:
            print(f"   ✓ Testing {dep_name}...")
            __import__(module_name)
            print(f"   ✅ {dep_name} import successful")
        except Exception as e:
            print(f"   ⚠️  {dep_name} import failed: {e}")
            print(f"      This may cause issues later. Try: pip install {dep_name}")

    return True


def test_environment():
    """Test environment variables and configuration"""
    print("\n⚙️  Testing environment configuration...")

    # Check .env file
    env_file = Path(".env")
    if env_file.exists():
        print("   ✅ .env file exists")

        # Check critical environment variables
        try:
            from mosaic.config.settings import MosaicSettings

            settings = MosaicSettings()

            missing = []
            if not settings.azure_openai_endpoint:
                missing.append("AZURE_OPENAI_ENDPOINT")
            if not settings.azure_cosmos_endpoint:
                missing.append("AZURE_COSMOS_DB_ENDPOINT")

            if missing:
                print(f"   ⚠️  Missing required environment variables: {missing}")
                print("      Please add these to your .env file")
                return False
            else:
                print("   ✅ Required environment variables are set")
                return True

        except Exception as e:
            print(f"   ❌ Environment configuration test failed: {e}")
            return False
    else:
        print("   ❌ .env file not found")
        print("      Copy .env.local to .env and add your Azure OpenAI credentials")
        return False


def test_basic_git_clone():
    """Test basic git cloning functionality"""
    print("\n🔗 Testing Git repository access...")

    try:
        import git
        import tempfile

        # Test cloning a small public repository
        with tempfile.TemporaryDirectory() as temp_dir:
            print("   ✓ Testing git clone with a small public repository...")

            # Clone a very small test repository
            repo_url = "https://github.com/octocat/Hello-World"  # Tiny test repo

            try:
                repo = git.Repo.clone_from(repo_url, temp_dir, depth=1)
                print(f"   ✅ Git clone successful: {repo.head.commit.hexsha[:8]}")
                return True
            except Exception as e:
                print(f"   ❌ Git clone failed: {e}")
                return False

    except Exception as e:
        print(f"   ❌ Git test setup failed: {e}")
        return False


async def test_ingestion_service_creation():
    """Test creating the ingestion service without full initialization"""
    print("\n🏗️  Testing IngestionService creation...")

    try:
        from mosaic.config.settings import MosaicSettings

        # Create settings with local config
        settings = MosaicSettings()
        print("   ✅ Settings created successfully")

        # Try to import IngestionPlugin
        from ingestion_service.plugins.ingestion import IngestionPlugin

        print("   ✅ IngestionPlugin imported successfully")

        # Create plugin instance without initialization
        plugin = IngestionPlugin(settings)
        print("   ✅ IngestionPlugin instance created successfully")

        # Test tree-sitter language initialization
        try:
            languages = plugin._initialize_languages()
            print(
                f"   ✅ Tree-sitter languages initialized: {len(languages)} languages"
            )
        except Exception as e:
            print(f"   ⚠️  Tree-sitter initialization failed: {e}")
            print(
                "      Some file parsing may not work, but basic functionality should still work"
            )

        print("   ✅ IngestionService creation test passed")
        return True

    except Exception as e:
        print(f"   ❌ IngestionService creation failed: {e}")
        traceback.print_exc()
        return False


async def main():
    """Run all tests"""
    print("🚀 Mosaic Ingestion Service - Basic Diagnostic Tests")
    print("=" * 60)

    all_passed = True

    # Run tests
    tests = [
        ("Import Tests", test_imports),
        ("Environment Tests", test_environment),
        ("Git Clone Tests", test_basic_git_clone),
        ("Service Creation Tests", test_ingestion_service_creation),
    ]

    results = {}
    for test_name, test_func in tests:
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            results[test_name] = result
            all_passed = all_passed and result
        except Exception as e:
            print(f"   ❌ {test_name} crashed: {e}")
            results[test_name] = False
            all_passed = False

    # Summary
    print("\n" + "=" * 60)
    print("📋 Test Results Summary:")
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status} {test_name}")

    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 All basic tests passed! The ingestion service should work.")
        print("\n📝 To run the full ingestion service:")
        print(
            "   python3 -m ingestion_service.main --repository-url https://github.com/ChrisMcKee1/Mosaic --branch main"
        )
    else:
        print(
            "⚠️  Some tests failed. Please fix the issues above before running the full service."
        )
        print("\n🔧 Common fixes:")
        print("   1. Install missing dependencies: pip install -r requirements.txt")
        print("   2. Copy .env.local to .env and add Azure OpenAI credentials")
        print(
            "   3. Ensure Docker containers are running: ./scripts/start-local-dev.sh"
        )

    return all_passed


if __name__ == "__main__":
    try:
        result = asyncio.run(main())
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test runner crashed: {e}")
        traceback.print_exc()
        sys.exit(1)
