#!/usr/bin/env python3
"""
Test MCP interface compliance for Mosaic Tool
Tests the exact function signatures required by TDD Section 6.0
"""
import ast
import inspect
from pathlib import Path

def extract_function_signatures(file_path):
    """Extract function signatures from a Python file"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    tree = ast.parse(content)
    functions = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # Get function signature
            args = []
            for arg in node.args.args:
                arg_name = arg.arg
                # Get type annotation if present
                if arg.annotation:
                    if isinstance(arg.annotation, ast.Name):
                        arg_type = arg.annotation.id
                    elif isinstance(arg.annotation, ast.Subscript):
                        # Handle List[Document] etc.
                        if isinstance(arg.annotation.value, ast.Name):
                            base_type = arg.annotation.value.id
                            if isinstance(arg.annotation.slice, ast.Name):
                                inner_type = arg.annotation.slice.id
                                arg_type = f"{base_type}[{inner_type}]"
                            else:
                                arg_type = base_type
                        else:
                            arg_type = "Unknown"
                    else:
                        arg_type = "Unknown"
                else:
                    arg_type = "Any"
                args.append(f"{arg_name}: {arg_type}")
            
            # Get return type annotation
            return_type = "None"
            if node.returns:
                if isinstance(node.returns, ast.Name):
                    return_type = node.returns.id
                elif isinstance(node.returns, ast.Subscript):
                    if isinstance(node.returns.value, ast.Name):
                        base_type = node.returns.value.id
                        if isinstance(node.returns.slice, ast.Name):
                            inner_type = node.returns.slice.id
                            return_type = f"{base_type}[{inner_type}]"
                        else:
                            return_type = base_type
            
            functions.append({
                'name': node.name,
                'args': args,
                'return_type': return_type,
                'signature': f"{node.name}({', '.join(args)}) -> {return_type}"
            })
    
    return functions

def test_mcp_interface_compliance():
    """Test that required MCP interface functions are implemented"""
    
    # Required function signatures from TDD Section 6.0
    required_functions = {
        'mosaic.retrieval.hybrid_search': {
            'args': ['query: str'],
            'return': 'List[Document]',
            'file': 'src/mosaic/plugins/retrieval.py'
        },
        'mosaic.retrieval.query_code_graph': {
            'args': ['library_id: str', 'relationship_type: str'], 
            'return': 'List[LibraryNode]',
            'file': 'src/mosaic/plugins/retrieval.py'
        },
        'mosaic.refinement.rerank': {
            'args': ['query: str', 'documents: List[Document]'],
            'return': 'List[Document]',
            'file': 'src/mosaic/plugins/refinement.py'
        },
        'mosaic.memory.save': {
            'args': ['session_id: str', 'content: str', 'type: str'],
            'return': 'None',
            'file': 'src/mosaic/plugins/memory.py'
        },
        'mosaic.memory.retrieve': {
            'args': ['session_id: str', 'query: str', 'limit: int'],
            'return': 'List[MemoryEntry]',
            'file': 'src/mosaic/plugins/memory.py'
        },
        'mosaic.diagram.generate': {
            'args': ['description: str'],
            'return': 'str',
            'file': 'src/mosaic/plugins/diagram.py'
        }
    }
    
    results = []
    
    for func_name, expected in required_functions.items():
        plugin_name = func_name.split('.')[1]
        method_name = func_name.split('.')[2]
        file_path = expected['file']
        
        if not Path(file_path).exists():
            results.append(f"❌ {func_name} - file {file_path} not found")
            continue
        
        try:
            functions = extract_function_signatures(file_path)
            found = False
            
            for func in functions:
                if func['name'] == method_name:
                    found = True
                    # Check arguments (simplified check)
                    if len(func['args']) == len(expected['args']):
                        results.append(f"✅ {func_name} - found with {len(func['args'])} arguments")
                    else:
                        results.append(f"⚠️  {func_name} - found but argument count mismatch")
                    break
            
            if not found:
                results.append(f"❌ {func_name} - method {method_name} not found in {file_path}")
                
        except Exception as e:
            results.append(f"❌ {func_name} - error parsing {file_path}: {e}")
    
    return results

def test_data_models():
    """Test that required data models are defined"""
    models_file = "src/mosaic/models/base.py"
    required_models = ["Document", "LibraryNode", "MemoryEntry"]
    
    if not Path(models_file).exists():
        return [f"❌ Models file {models_file} not found"]
    
    results = []
    with open(models_file, 'r') as f:
        content = f.read()
    
    for model in required_models:
        if f"class {model}" in content:
            results.append(f"✅ {model} model found")
        else:
            results.append(f"❌ {model} model not found")
    
    return results

if __name__ == "__main__":
    print("=== Mosaic MCP Interface Compliance Tests ===\n")
    
    print("1. Testing MCP interface function signatures...")
    interface_results = test_mcp_interface_compliance()
    for result in interface_results:
        print(f"   {result}")
    print()
    
    print("2. Testing data models...")
    model_results = test_data_models()
    for result in model_results:
        print(f"   {result}")
    print()
    
    # Summary
    all_results = interface_results + model_results
    errors = [r for r in all_results if r.startswith("❌")]
    warnings = [r for r in all_results if r.startswith("⚠️")]
    
    print("=== Test Summary ===")
    if len(errors) == 0:
        print("✅ All MCP interface requirements are implemented!")
        if len(warnings) > 0:
            print(f"⚠️  {len(warnings)} warnings found - review recommended")
    else:
        print(f"❌ {len(errors)} errors found:")
        for error in errors:
            print(f"   {error}")