---
description: Safe Azure deployment with validation and tracking
mode: agent
---

# Azure Deployment Workflow

Safe Azure deployment process with comprehensive validation and rollback capabilities.

## Input

Deployment Description: ${input:deployment:Describe what you're deploying (e.g., Production deployment of Mosaic MCP Tool v0.1.0)}

## Workflow Steps

### 1. Pre-Deployment Planning

- **Environment Verification**: Confirm target environment (dev/staging/prod)
- **Resource Review**: Verify all required Azure resources are available
- **Dependency Check**: Ensure all dependencies are properly configured
- **Backup Strategy**: Plan backup and rollback procedures

### 2. Infrastructure Validation

- **Bicep Template Validation**: Validate ARM/Bicep templates
- **Resource Group Check**: Verify resource group and permissions
- **Network Configuration**: Validate network security groups and connectivity
- **Service Principal Rights**: Confirm deployment service principal has required permissions

### 3. Pre-Deployment Testing

- **Template Deployment Test**: Run template validation in Azure
- **Connectivity Tests**: Verify service endpoints and health checks
- **Configuration Validation**: Confirm all environment variables and settings
- **Security Scan**: Run security validation on deployment artifacts

### 4. Deployment Execution

- **Infrastructure Deployment**: Deploy Azure resources using Bicep templates
- **Application Deployment**: Deploy application code to Container Apps
- **Configuration Updates**: Apply environment-specific configuration
- **Service Startup**: Start services and verify initial health

### 5. Post-Deployment Validation

- **Health Checks**: Verify all services are healthy and responding
- **Functional Testing**: Run critical path tests to ensure functionality
- **Performance Validation**: Check response times and resource utilization
- **Security Verification**: Validate security configurations are applied

### 6. Monitoring and Documentation

- **Enable Monitoring**: Ensure Application Insights and logging are active
- **Alert Configuration**: Set up alerts for critical metrics
- **ConPort Updates**: Log deployment details and any issues encountered
- **Documentation**: Update deployment runbooks and troubleshooting guides

## Deployment Checklist

- [ ] Backup current state
- [ ] Validate Bicep templates
- [ ] Test connectivity
- [ ] Deploy infrastructure
- [ ] Deploy application
- [ ] Run health checks
- [ ] Verify functionality
- [ ] Enable monitoring
- [ ] Document deployment

## Rollback Procedures

If deployment issues occur:

1. Stop new traffic to problematic services
2. Revert to previous application version
3. Restore previous infrastructure state if needed
4. Validate rollback success
5. Document issues for post-mortem analysis
