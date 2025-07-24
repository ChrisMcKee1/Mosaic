# 🔐 Mosaic Security Configuration Setup

## ⚠️ **CRITICAL: Environment Variables and Secrets Management**

### 🚨 **Security Model:**

1. **Templates in Git** (✅ Safe) - `.env.*.template` files with placeholder values
2. **Actual Config Files Ignored** (🔒 Secure) - Real `.env.*` files never committed  
3. **Secrets in External Systems** (🛡️ Production) - Azure Key Vault, GitHub Secrets

### 📋 **Setup Instructions:**

#### **Step 1: Copy Template Files**
```bash
# Copy templates to actual config files
cp .env.shared.template .env.shared
cp .env.mosaic.local.template .env.mosaic.local
cp .env.ingestion_service.local.template .env.ingestion_service.local
cp .env.mosaic-ui.local.template .env.mosaic-ui.local
```

#### **Step 2: Fill in Real Values**
Edit the copied `.env.*` files with your actual:
- Azure OpenAI endpoint and API keys
- Azure Cosmos DB connection strings
- Microsoft Entra ID tenant/client IDs
- Azure subscription and resource group info

#### **Step 3: Verify .gitignore Protection**
```bash
# These files should NEVER be committed:
git status --ignored | grep .env.
```

### 🏗️ **Configuration Hierarchy:**

```
Environment Variable Priority (highest to lowest):
1. Runtime Environment Variables (Azure, CLI)
2. .env.{service}.local (service-specific overrides)  
3. .env.shared (common base configuration)
4. Default values in code
```

### 🛡️ **Production Security:**

- **Azure**: Uses Managed Identity + Key Vault
- **GitHub Actions**: Uses GitHub Secrets
- **Local Development**: Uses `.env.*` files (gitignored)

### ❌ **What NOT to Do:**

- Never commit files with actual API keys, connection strings, or passwords
- Never push `.env.*` files (without `.template` suffix) to git
- Never hardcode secrets in source code
- Never share actual environment files via chat, email, or other channels

### ✅ **What's Safe to Commit:**

- `.env.*.template` files with placeholder/example values
- Configuration code that loads from environment variables
- Documentation about configuration structure
- Infrastructure templates (Bicep, Terraform) that reference variables

### 🔍 **Why Local Development Worked:**

The services worked locally because:
1. **Graceful degradation**: Code falls back to simulated mode when secrets missing
2. **Local-only operations**: Some services (like local ingestion) don't need cloud APIs
3. **Mock data**: Streamlit app shows static data when backend unavailable

For full functionality, you need real Azure service credentials.