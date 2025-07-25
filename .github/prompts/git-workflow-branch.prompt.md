---
description: Create and manage git branches with proper tracking
mode: agent
---

# Git Branch Workflow

Create a new git branch with proper naming conventions and tracking setup.

## Input

Branch Name: ${input:branch_name:Enter the branch name (e.g., feature/FR-8-semantic-reranking)}

## Workflow Steps

### 1. Planning and Validation

- Validate branch name follows convention: `feature/FR-X-description` or `bugfix/description`
- Check if branch already exists locally or remotely
- Ensure working directory is clean

### 2. Branch Creation

```bash
git checkout main
git pull
git checkout -b "${branch_name}"
```

### 3. Documentation and Tracking

- Create branch tracking in ConPort if related to a task
- Update local documentation about the branch purpose
- Set up any environment-specific configuration needed

### 4. Environment Setup

- Verify virtual environment is activated
- Install any new dependencies if needed
- Run initial validation to ensure environment is ready

### 5. ConPort Integration

- Link branch to related ConPort task if applicable
- Log branch creation decision in ConPort
- Set up progress tracking for the branch work

## Branch Naming Conventions

- **Feature branches**: `feature/FR-X-short-description`
- **Bug fixes**: `bugfix/issue-description`
- **Hotfixes**: `hotfix/critical-issue`
- **Documentation**: `docs/topic-description`
- **Infrastructure**: `infra/infrastructure-change`
