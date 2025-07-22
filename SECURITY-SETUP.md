# Security Setup Guide

## Overview

This project follows security best practices by externalizing all sensitive configuration values to environment variables. **No sensitive information should ever be committed to source control.**

## 🔐 Security Measures Implemented

### 1. Externalized Configuration
- All sensitive values (API keys, connection strings, passwords) are read from environment variables
- No hardcoded secrets in source code
- Configuration logic is versioned, but sensitive values are not

### 2. Environment File Structure
- `.env` files are ignored by git (see `.gitignore`)
- `environment-variables.template` provides a template for required variables
- Local development uses safe defaults where possible

### 3. Validation
- Configuration validation ensures required environment variables are set
- Clear error messages guide developers to missing configuration

## 🚀 Quick Setup for Local Development

### Step 1: Create Environment File
```bash
# Copy the template to create your local environment file
cp environment-variables.template .env
```

### Step 2: Update Values
Edit `.env` and update the placeholder values:

```bash
# Required: Update these with your actual Azure OpenAI credentials
AZURE_OPENAI_ENDPOINT=https://your-actual-openai-resource.openai.azure.com/
AZURE_TENANT_ID=your-actual-tenant-id
AZURE_CLIENT_ID=your-actual-client-id
AZURE_CLIENT_SECRET=your-actual-client-secret

# Optional: Change local development passwords
REDIS_PASSWORD=your-secure-local-password
```

### Step 3: Start Local Services
```bash
# Start local emulators (Cosmos DB, Redis, Azurite)
docker-compose -f docker-compose.local.yml up -d
```

### Step 4: Verify Configuration
```bash
# Test that configuration loads correctly
python -c "from src.mosaic.config.local_config import local_config; print('✅ Configuration loaded successfully')"
```

## 🔍 Environment Variables Reference

### Required for All Environments
- `AZURE_OPENAI_ENDPOINT` - Your Azure OpenAI service endpoint
- `MOSAIC_COSMOS_KEY` - Cosmos DB access key
- `REDIS_PASSWORD` - Redis authentication password
- `AZURE_STORAGE_CONNECTION_STRING` - Storage account connection string

### Local Development Defaults
- Cosmos DB Emulator key is provided in template (safe for local use)
- Azurite connection string is provided in template (safe for local use)
- Redis runs on localhost with custom password

### Production Considerations
- Use Azure Managed Identity when possible (`AZURE_USE_MANAGED_IDENTITY=true`)
- Store secrets in Azure Key Vault
- Use Azure Container Apps environment variables for deployment
- Never commit production credentials to source control

## ⚠️ Security Warnings

### DO NOT:
- ❌ Commit `.env` files to git
- ❌ Share environment files containing real credentials
- ❌ Use production credentials in development
- ❌ Hardcode secrets in source code

### DO:
- ✅ Use the environment variable template
- ✅ Store production secrets in Azure Key Vault
- ✅ Use different credentials for each environment
- ✅ Rotate credentials regularly

## 🛠️ Troubleshooting

### Configuration Validation Errors
If you see errors about missing configuration:

1. Check that your `.env` file exists and contains all required variables
2. Verify that variable names match exactly (case-sensitive)
3. Ensure no extra spaces around `=` in environment assignments
4. Check that the `.env` file is in the project root directory

### Local Emulator Issues
If local services fail to connect:

1. Ensure Docker services are running: `docker-compose -f docker-compose.local.yml ps`
2. Check that ports 8081 (Cosmos), 6379 (Redis), and 10000-10002 (Azurite) are available
3. Verify SSL settings for local Cosmos DB emulator

### Azure Service Connection Issues
If Azure services fail to authenticate:

1. Verify your Azure OpenAI endpoint is correct and accessible
2. Check that your service principal has necessary permissions
3. Ensure your tenant/client IDs are correct
4. Test credentials using Azure CLI: `az login --service-principal`

## 📞 Getting Help

If you encounter security-related issues:

1. Check this guide first
2. Review the `environment-variables.template` for required variables
3. Verify your `.env` file against the template
4. Check Azure service permissions and credentials
5. Open an issue if problems persist (never include actual credentials!)

## 🔄 Rotating Credentials

For production environments, establish a credential rotation schedule:

1. **Azure OpenAI Keys**: Rotate monthly via Azure Portal
2. **Service Principal Secrets**: Rotate quarterly 
3. **Cosmos DB Keys**: Rotate bi-annually
4. **Redis Password**: Rotate monthly for production instances

Remember to update all deployment environments when rotating credentials. 