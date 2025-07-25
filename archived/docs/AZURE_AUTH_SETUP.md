# Azure Authentication Setup for Mosaic MCP Tool

## Quick Setup for End-to-End Testing

### 1. Create Local Configuration Files

Copy the templates and fill with your actual Azure credentials:

```bash
# Copy shared configuration template
cp .env.shared.template .env.shared

# Copy service-specific configuration templates  
cp .env.mosaic.local.template .env.mosaic.local
cp .env.ingestion_service.local.template .env.ingestion_service.local
cp .env.mosaic-ui.local.template .env.mosaic-ui.local
```

### 2. Fill in Real Azure Credentials

Edit `.env.shared` with your actual values:

```bash
# Azure OpenAI Service (Required for LLM and embedding operations)
AZURE_OPENAI_ENDPOINT=https://your-openai-service.openai.azure.com/
AZURE_OPENAI_API_KEY=your-actual-api-key-here
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME=gpt-4
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME=text-embedding-ada-002

# Azure Cosmos DB (Required for OmniRAG graph database)
AZURE_COSMOS_ENDPOINT=https://your-cosmos-account.documents.azure.com:443/
AZURE_COSMOS_KEY=your-actual-cosmos-key-here

# Azure authentication (for production OAuth)
AZURE_TENANT_ID=your-tenant-id
AZURE_CLIENT_ID=your-client-id  
AZURE_CLIENT_SECRET=your-client-secret

# Local development mode
AZURE_USE_MANAGED_IDENTITY=false
DEBUG=true
ENVIRONMENT=development
```

### 3. Where to Get These Values

**Azure OpenAI Service:**
1. Go to Azure Portal → Azure OpenAI Service
2. Copy endpoint URL and API key from "Keys and Endpoint" section
3. Note your model deployment names

**Azure Cosmos DB:**
1. Go to Azure Portal → Azure Cosmos DB account
2. Copy endpoint URL and primary key from "Keys" section

**Azure Active Directory:**
1. Go to Azure Portal → Azure Active Directory → App registrations
2. Create or select your app registration
3. Copy Tenant ID, Client ID, and create/copy Client Secret

### 4. Test Authentication

```bash
# Test Azure CLI authentication
az login
az account show

# Test with Python
python -c "
from azure.identity import DefaultAzureCredential
from azure.cosmos import CosmosClient
cred = DefaultAzureCredential()
print('Authentication successful')
"
```

### 5. Security Notes

- **Local files** (`.env.shared`, `.env.*.local`) contain real secrets - never commit to git
- **Template files** (`.env.*.template`) are safe placeholders - committed to git  
- **Production** uses Azure Managed Identity (no API keys needed)
- **.gitignore** already excludes all `.env.*` files except templates

### 6. Why Development "Worked" Without Real Credentials

The services use Azure's `DefaultAzureCredential()` which gracefully falls back:

1. **With real credentials**: Connects to actual Azure services
2. **Without credentials**: Runs in "simulated mode" with mock data
3. **In production**: Uses Azure Managed Identity automatically

This allows development to proceed even without full Azure setup, but for end-to-end testing you need the real credentials above.