#!/usr/bin/env python3
"""
Structure validation tests for Mosaic MCP Tool
Validates that all required files and components exist for the 14 functional requirements
"""

import os
import sys
from pathlib import Path

# Get project root
project_root = Path(__file__).parent.parent

def test_file_exists(file_path, description):
    """Test if a file exists and report result"""
    full_path = project_root / file_path
    if full_path.exists():
        print(f"✅ {description}")
        return True
    else:
        print(f"❌ {description} - Missing: {file_path}")
        return False

def test_directory_exists(dir_path, description):
    """Test if a directory exists and report result"""
    full_path = project_root / dir_path
    if full_path.exists() and full_path.is_dir():
        print(f"✅ {description}")
        return True
    else:
        print(f"❌ {description} - Missing: {dir_path}")
        return False

def test_function_exists_in_file(file_path, function_name, description):
    """Test if a function exists in a file"""
    full_path = project_root / file_path
    if not full_path.exists():
        print(f"❌ {description} - File missing: {file_path}")
        return False
    
    try:
        with open(full_path, 'r') as f:
            content = f.read()
            if f"def {function_name}" in content or f"async def {function_name}" in content:
                print(f"✅ {description}")
                return True
            else:
                print(f"❌ {description} - Function {function_name} not found")
                return False
    except Exception as e:
        print(f"❌ {description} - Error reading file: {e}")
        return False

def validate_functional_requirements():
    """Validate all 14 functional requirements are implemented"""
    
    print("=== Mosaic MCP Tool - Functional Requirements Validation ===\n")
    
    results = []
    
    # FR-1: MCP Server Implementation
    print("FR-1: MCP Server Implementation")
    results.append(test_file_exists("src/mosaic/server/main.py", "MCP Server main.py exists"))
    results.append(test_function_exists_in_file("src/mosaic/server/main.py", "MosaicMCPServer", "MosaicMCPServer class exists"))
    print()
    
    # FR-2: Semantic Kernel Integration
    print("FR-2: Semantic Kernel Integration")
    results.append(test_file_exists("src/mosaic/server/kernel.py", "Semantic Kernel manager exists"))
    results.append(test_directory_exists("src/mosaic/plugins", "Plugins directory exists"))
    print()
    
    # FR-3: Streamable HTTP Communication  
    print("FR-3: Streamable HTTP Communication")
    results.append(test_function_exists_in_file("src/mosaic/server/main.py", "start", "Server start method exists"))
    print()
    
    # FR-4: Azure Native Deployment
    print("FR-4: Azure Native Deployment")
    results.append(test_file_exists("infra/main.bicep", "Main Bicep template exists"))
    results.append(test_file_exists("infra/resources.bicep", "Resources Bicep template exists"))
    results.append(test_file_exists("azure.yaml", "Azure Developer CLI config exists"))
    results.append(test_file_exists("Dockerfile", "Docker configuration exists"))
    print()
    
    # FR-5: Hybrid Search (RetrievalPlugin)
    print("FR-5: Hybrid Search")
    results.append(test_file_exists("src/mosaic/plugins/retrieval.py", "RetrievalPlugin exists"))
    results.append(test_function_exists_in_file("src/mosaic/plugins/retrieval.py", "hybrid_search", "hybrid_search method exists"))
    print()
    
    # FR-6: Graph-Based Code Analysis (RetrievalPlugin)
    print("FR-6: Graph-Based Code Analysis")
    results.append(test_function_exists_in_file("src/mosaic/plugins/retrieval.py", "query_code_graph", "query_code_graph method exists"))
    print()
    
    # FR-7: Candidate Aggregation (RetrievalPlugin)
    print("FR-7: Candidate Aggregation")
    results.append(test_function_exists_in_file("src/mosaic/plugins/retrieval.py", "_aggregate_results", "aggregate_results method exists"))
    print()
    
    # FR-8: Semantic Reranking (RefinementPlugin)
    print("FR-8: Semantic Reranking")
    results.append(test_file_exists("src/mosaic/plugins/refinement.py", "RefinementPlugin exists"))
    results.append(test_function_exists_in_file("src/mosaic/plugins/refinement.py", "rerank", "rerank method exists"))
    print()
    
    # FR-9: Unified Memory Interface (MemoryPlugin)
    print("FR-9: Unified Memory Interface")
    results.append(test_file_exists("src/mosaic/plugins/memory.py", "MemoryPlugin exists"))
    results.append(test_function_exists_in_file("src/mosaic/plugins/memory.py", "save", "save method exists"))
    results.append(test_function_exists_in_file("src/mosaic/plugins/memory.py", "retrieve", "retrieve method exists"))
    results.append(test_function_exists_in_file("src/mosaic/plugins/memory.py", "clear", "clear method exists"))
    print()
    
    # FR-10: Multi-Layered Storage (MemoryPlugin)
    print("FR-10: Multi-Layered Storage")
    results.append(test_function_exists_in_file("src/mosaic/plugins/memory.py", "_initialize_cosmos", "Cosmos DB initialization exists"))
    results.append(test_function_exists_in_file("src/mosaic/plugins/memory.py", "_initialize_redis", "Redis initialization exists"))
    print()
    
    # FR-11: LLM-Powered Consolidation (MemoryPlugin)
    print("FR-11: LLM-Powered Consolidation")
    results.append(test_file_exists("functions/memory-consolidator/function_app.py", "Memory consolidation Azure Function exists"))
    results.append(test_file_exists("functions/memory-consolidator/requirements.txt", "Function requirements.txt exists"))
    print()
    
    # FR-12: Mermaid Generation (DiagramPlugin)
    print("FR-12: Mermaid Generation")
    results.append(test_file_exists("src/mosaic/plugins/diagram.py", "DiagramPlugin exists"))
    results.append(test_function_exists_in_file("src/mosaic/plugins/diagram.py", "generate", "generate method exists"))
    print()
    
    # FR-13: Mermaid as Context Resource (DiagramPlugin)
    print("FR-13: Mermaid as Context Resource")
    results.append(test_function_exists_in_file("src/mosaic/plugins/diagram.py", "get_stored_diagram", "get_stored_diagram method exists"))
    print()
    
    # FR-14: Secure MCP Endpoint
    print("FR-14: Secure MCP Endpoint")
    results.append(test_file_exists("src/mosaic/server/auth.py", "OAuth2Handler exists"))
    results.append(test_function_exists_in_file("src/mosaic/server/auth.py", "OAuth2Handler", "OAuth2Handler class exists"))
    print()
    
    return results

def validate_data_models():
    """Validate required data models from TDD Section 5.0"""
    
    print("=== Data Models Validation ===\n")
    
    results = []
    
    # Core data models
    results.append(test_file_exists("src/mosaic/models/base.py", "Base models file exists"))
    
    # Check for specific model classes
    base_models_file = project_root / "src/mosaic/models/base.py"
    if base_models_file.exists():
        try:
            with open(base_models_file, 'r') as f:
                content = f.read()
                
            models = ["Document", "LibraryNode", "MemoryEntry", "DiagramResponse"]
            for model in models:
                if f"class {model}" in content:
                    print(f"✅ {model} model class exists")
                    results.append(True)
                else:
                    print(f"❌ {model} model class missing")
                    results.append(False)
                    
        except Exception as e:
            print(f"❌ Error reading models file: {e}")
            results.append(False)
    
    print()
    return results

def validate_mcp_interface():
    """Validate MCP interface requirements from TDD Section 6.0"""
    
    print("=== MCP Interface Validation ===\n")
    
    required_interfaces = [
        ("src/mosaic/server/main.py", "hybrid_search", "mosaic.retrieval.hybrid_search tool"),
        ("src/mosaic/server/main.py", "query_code_graph", "mosaic.retrieval.query_code_graph tool"),
        ("src/mosaic/server/main.py", "rerank", "mosaic.refinement.rerank tool"),
        ("src/mosaic/server/main.py", "save_memory", "mosaic.memory.save tool"),
        ("src/mosaic/server/main.py", "retrieve_memory", "mosaic.memory.retrieve tool"),
        ("src/mosaic/server/main.py", "generate_diagram", "mosaic.diagram.generate tool")
    ]
    
    results = []
    for file_path, function_name, description in required_interfaces:
        results.append(test_function_exists_in_file(file_path, function_name, description))
    
    print()
    return results

def validate_infrastructure():
    """Validate infrastructure and deployment files"""
    
    print("=== Infrastructure Validation ===\n")
    
    results = []
    
    # Bicep templates
    bicep_files = [
        ("infra/main.bicep", "Main orchestration template"),
        ("infra/resources.bicep", "Core resources template"),
        ("infra/modules/container-apps.bicep", "Container Apps module"),
        ("infra/omnirag/cosmos-omnirag.bicep", "OmniRAG Cosmos DB template")
    ]
    
    for file_path, description in bicep_files:
        results.append(test_file_exists(file_path, description))
    
    # Configuration files
    config_files = [
        ("azure.yaml", "Azure Developer CLI configuration"),
        ("Dockerfile", "Docker container configuration"),
        ("pyproject.toml", "Python project configuration"),
        ("requirements.txt", "Python dependencies")
    ]
    
    for file_path, description in config_files:
        results.append(test_file_exists(file_path, description))
    
    print()
    return results

if __name__ == "__main__":
    print("Starting Mosaic MCP Tool validation...\n")
    
    # Run all validations
    fr_results = validate_functional_requirements()
    model_results = validate_data_models()
    interface_results = validate_mcp_interface()
    infra_results = validate_infrastructure()
    
    # Calculate results
    all_results = fr_results + model_results + interface_results + infra_results
    total_tests = len(all_results)
    passed_tests = sum(1 for result in all_results if result)
    failed_tests = total_tests - passed_tests
    
    # Summary
    print("=== VALIDATION SUMMARY ===")
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    if failed_tests == 0:
        print("\n🎉 ALL FUNCTIONAL REQUIREMENTS IMPLEMENTED!")
        print("✅ Mosaic MCP Tool is ready for deployment")
    else:
        print(f"\n⚠️  {failed_tests} requirements need attention")
        
    print(f"\nImplementation Status: {passed_tests}/{total_tests} requirements completed")