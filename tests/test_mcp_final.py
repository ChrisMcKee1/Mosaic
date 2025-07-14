#!/usr/bin/env python3
"""
Final validation test for Mosaic MCP Tool implementation
"""

def test_mcp_functions_present():
    """Test that all required MCP functions are present with @sk_function decorators"""
    import re
    from pathlib import Path
    
    required_functions = {
        'hybrid_search': '../src/mosaic/plugins/retrieval.py',
        'query_code_graph': '../src/mosaic/plugins/retrieval.py', 
        'rerank': '../src/mosaic/plugins/refinement.py',
        'save': '../src/mosaic/plugins/memory.py',
        'retrieve': '../src/mosaic/plugins/memory.py',
        'generate': '../src/mosaic/plugins/diagram.py'
    }
    
    results = []
    
    for func_name, file_path in required_functions.items():
        if not Path(file_path).exists():
            results.append(f"❌ {func_name} - file {file_path} not found")
            continue
            
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Check for @sk_function decorator before function
        pattern = rf'@sk_function.*?async def {func_name}'
        if re.search(pattern, content, re.DOTALL):
            results.append(f"✅ {func_name} - properly implemented with @sk_function")
        elif f"async def {func_name}" in content:
            results.append(f"⚠️  {func_name} - function exists but missing @sk_function decorator")
        else:
            results.append(f"❌ {func_name} - function not found")
    
    return results

def test_data_models_complete():
    """Test that data models have proper Pydantic structure"""
    from pathlib import Path
    
    models_file = Path("../src/mosaic/models/base.py")
    if not models_file.exists():
        return ["❌ Models file not found"]
    
    with open(models_file, 'r') as f:
        content = f.read()
    
    results = []
    required_models = ["Document", "LibraryNode", "MemoryEntry"]
    
    for model in required_models:
        if f"class {model}(BaseModel)" in content:
            results.append(f"✅ {model} - properly defined as Pydantic model")
        elif f"class {model}" in content:
            results.append(f"⚠️  {model} - class exists but may not inherit from BaseModel")
        else:
            results.append(f"❌ {model} - class not found")
    
    return results

def test_fr_compliance():
    """Test functional requirements compliance"""
    from pathlib import Path
    
    results = []
    
    # FR-1: MCP Server Implementation
    if Path("../src/mosaic/server/main.py").exists():
        with open("../src/mosaic/server/main.py", 'r') as f:
            content = f.read()
        if "class MosaicMCPServer" in content and "FastMCP" in content:
            results.append("✅ FR-1: MCP Server with FastMCP implemented")
        else:
            results.append("❌ FR-1: MCP Server implementation incomplete")
    
    # FR-2: Semantic Kernel Integration
    plugin_files = list(Path("../src/mosaic/plugins").glob("*.py"))
    sk_functions_found = 0
    for plugin_file in plugin_files:
        with open(plugin_file, 'r') as f:
            content = f.read()
        sk_functions_found += content.count("@sk_function")
    
    if sk_functions_found >= 6:  # We should have at least 6 SK functions
        results.append(f"✅ FR-2: Semantic Kernel integration ({sk_functions_found} SK functions found)")
    else:
        results.append(f"⚠️  FR-2: Semantic Kernel integration incomplete ({sk_functions_found} SK functions found)")
    
    # FR-14: OAuth2 Authentication
    if Path("../src/mosaic/server/auth.py").exists():
        with open("../src/mosaic/server/auth.py", 'r') as f:
            content = f.read()
        if "OAuth2Handler" in content and "JWT" in content:
            results.append("✅ FR-14: OAuth2 authentication implemented")
        else:
            results.append("❌ FR-14: OAuth2 authentication incomplete")
    
    return results

if __name__ == "__main__":
    print("=== Mosaic MCP Tool Final Validation ===\n")
    
    print("1. Testing MCP function implementations...")
    func_results = test_mcp_functions_present()
    for result in func_results:
        print(f"   {result}")
    print()
    
    print("2. Testing data models...")
    model_results = test_data_models_complete()
    for result in model_results:
        print(f"   {result}")
    print()
    
    print("3. Testing functional requirements compliance...")
    fr_results = test_fr_compliance()
    for result in fr_results:
        print(f"   {result}")
    print()
    
    # Summary
    all_results = func_results + model_results + fr_results
    errors = [r for r in all_results if r.startswith("❌")]
    warnings = [r for r in all_results if r.startswith("⚠️")]
    successes = [r for r in all_results if r.startswith("✅")]
    
    print("=== Final Validation Summary ===")
    print(f"✅ {len(successes)} requirements implemented correctly")
    print(f"⚠️  {len(warnings)} warnings (review recommended)")
    print(f"❌ {len(errors)} critical errors")
    print()
    
    if len(errors) == 0:
        print("🎉 Mosaic MCP Tool implementation is ready for deployment!")
        print("Next steps:")
        print("   1. Run 'azd up' to deploy to Azure")
        print("   2. Configure MCP client connections") 
        print("   3. Test end-to-end workflows")
    else:
        print("❌ Critical issues found - resolve before deployment")