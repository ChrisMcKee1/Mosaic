#!/usr/bin/env python3
"""
Basic test to validate core implementation structure
"""
import sys
import os
import ast
from pathlib import Path

def test_syntax_validity():
    """Test that all Python files have valid syntax"""
    src_dir = Path("../src/mosaic")
    errors = []
    
    for py_file in src_dir.rglob("*.py"):
        try:
            with open(py_file, 'r') as f:
                content = f.read()
            ast.parse(content)
            print(f"✅ {py_file} - syntax valid")
        except SyntaxError as e:
            errors.append(f"❌ {py_file} - syntax error: {e}")
        except Exception as e:
            errors.append(f"❌ {py_file} - error: {e}")
    
    return errors

def test_required_files():
    """Test that all required files exist"""
    required_files = [
        "../src/mosaic/config/settings.py",
        "../src/mosaic/server/main.py", 
        "../src/mosaic/server/kernel.py",
        "../src/mosaic/server/auth.py",
        "../src/mosaic/plugins/retrieval.py",
        "../src/mosaic/plugins/refinement.py",
        "../src/mosaic/plugins/memory.py",
        "../src/mosaic/plugins/diagram.py",
        "../src/mosaic/models/__init__.py"
    ]
    
    missing = []
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path} - exists")
        else:
            missing.append(f"❌ {file_path} - missing")
    
    return missing

def test_class_definitions():
    """Test that required classes are defined"""
    from pathlib import Path
    
    # Test for MosaicMCPServer class
    main_py = Path("../src/mosaic/server/main.py")
    if main_py.exists():
        with open(main_py) as f:
            content = f.read()
            if "class MosaicMCPServer" in content:
                print("✅ MosaicMCPServer class found")
            else:
                print("❌ MosaicMCPServer class not found")
    
    # Test for plugin classes
    plugins = ["RetrievalPlugin", "RefinementPlugin", "MemoryPlugin", "DiagramPlugin"]
    for plugin in plugins:
        plugin_file = Path(f"../src/mosaic/plugins/{plugin.lower().replace('plugin', '')}.py")
        if plugin_file.exists():
            with open(plugin_file) as f:
                content = f.read()
                if f"class {plugin}" in content:
                    print(f"✅ {plugin} class found")
                else:
                    print(f"❌ {plugin} class not found")

if __name__ == "__main__":
    print("=== Mosaic MCP Tool Basic Tests ===\n")
    
    print("1. Testing syntax validity...")
    syntax_errors = test_syntax_validity()
    print()
    
    print("2. Testing required files...")
    missing_files = test_required_files()
    print()
    
    print("3. Testing class definitions...")
    test_class_definitions()
    print()
    
    # Summary
    print("=== Test Summary ===")
    total_errors = len(syntax_errors) + len(missing_files)
    
    if total_errors == 0:
        print("✅ All basic tests passed!")
        sys.exit(0)
    else:
        print(f"❌ {total_errors} errors found:")
        for error in syntax_errors + missing_files:
            print(f"   {error}")
        sys.exit(1)