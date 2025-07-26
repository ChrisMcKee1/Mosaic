#!/usr/bin/env python3
"""
Simple test script to debug Cosmos DB emulator connection.
"""

import os
import urllib3
import requests
from azure.cosmos import CosmosClient

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Local emulator configuration
ENDPOINT = "https://localhost:8081"
KEY = "C2y6yDjf5/R+ob0N8A7Cgv30VRDJIWEHLM+4QDU5DE2nQ9nDuVTqobD4b8mGGyPMbIZnqyMsEcaGQy67XIw/Jw=="

print("🔍 Testing Cosmos DB Emulator Connection...")
print(f"📍 Endpoint: {ENDPOINT}")

# Test 1: Simple HTTP request to check if emulator is responding
print("\n1️⃣ Testing basic HTTP connectivity...")
try:
    response = requests.get(ENDPOINT, verify=False, timeout=10)
    print(f"✅ HTTP Status: {response.status_code}")
    print(f"📄 Response headers: {dict(response.headers)}")
except Exception as e:
    print(f"❌ HTTP test failed: {e}")

# Test 2: Test with minimal Cosmos client
print("\n2️⃣ Testing minimal Cosmos client...")
try:
    # Set environment variables to disable SSL verification
    os.environ["PYTHONHTTPSVERIFY"] = "0"
    os.environ["CURL_CA_BUNDLE"] = ""

    client = CosmosClient(ENDPOINT, KEY)
    print("✅ Cosmos client created successfully")

    # Try to get account info
    print("3️⃣ Testing account info...")
    databases = list(client.list_databases())
    print(f"✅ Found {len(databases)} databases:")
    for db in databases:
        print(f"   • {db['id']}")

except Exception as e:
    print(f"❌ Cosmos client test failed: {e}")
    import traceback

    traceback.print_exc()

print("\n🏁 Test completed!")
