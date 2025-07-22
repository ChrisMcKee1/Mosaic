"""
Comprehensive test suite for AST parsing and entity extraction.

Tests the enhanced tree-sitter implementation across all 11 supported languages
with various code patterns, error conditions, and edge cases.
"""

import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

# Import the ingestion plugin
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

class TestASTParsingComprehensive:
    """Comprehensive test suite for AST parsing across 11 languages."""
    
    @pytest.fixture
    def ingestion_plugin(self):
        """Create actual ingestion plugin instance."""
        from ingestion_service.plugins.ingestion import IngestionPlugin
        return IngestionPlugin()
    
    @pytest.fixture
    def temp_files(self):
        """Create temporary files for testing."""
        temp_dir = tempfile.mkdtemp()
        files = {}
        
        # Python test files
        files['simple.py'] = '''
def hello_world():
    """A simple function."""
    print("Hello, World!")

class Calculator:
    """A simple calculator class."""
    
    def add(self, a, b):
        return a + b
    
    async def async_calculate(self, x):
        return x * 2

from math import sqrt
import os
'''
        
        # JavaScript test files
        files['app.js'] = '''
function greet(name) {
    console.log(`Hello, ${name}!`);
}

const multiply = (a, b) => a * b;

class User {
    constructor(name) {
        this.name = name;
    }
    
    getName() {
        return this.name;
    }
}

import { Component } from 'react';
export default User;
'''
        
        # TypeScript test files
        files['types.ts'] = '''
interface Person {
    name: string;
    age: number;
}

type UserRole = 'admin' | 'user';

function createUser(person: Person): User {
    return new User(person.name);
}

export class UserService {
    private users: Person[] = [];
    
    addUser(user: Person): void {
        this.users.push(user);
    }
}
'''
        
        # Java test files
        files['Example.java'] = '''
package com.example;

import java.util.List;
import java.util.ArrayList;

public class Example {
    private List<String> items;
    
    public Example() {
        this.items = new ArrayList<>();
    }
    
    public void addItem(String item) {
        items.add(item);
    }
    
    public interface ItemProcessor {
        void process(String item);
    }
}
'''
        
        # Go test files
        files['main.go'] = '''
package main

import (
    "fmt"
    "net/http"
)

type Server struct {
    port int
}

func (s *Server) Start() error {
    return http.ListenAndServe(fmt.Sprintf(":%d", s.port), nil)
}

func NewServer(port int) *Server {
    return &Server{port: port}
}

const DefaultPort = 8080
var globalServer *Server

interface Handler {
    Handle(w http.ResponseWriter, r *http.Request)
}
'''
        
        # Rust test files
        files['lib.rs'] = '''
use std::collections::HashMap;

pub struct Cache {
    data: HashMap<String, String>,
}

impl Cache {
    pub fn new() -> Self {
        Cache {
            data: HashMap::new(),
        }
    }
    
    pub fn get(&self, key: &str) -> Option<&String> {
        self.data.get(key)
    }
}

pub trait Storage {
    fn store(&mut self, key: String, value: String);
}

pub enum Status {
    Active,
    Inactive,
}

pub const MAX_SIZE: usize = 1000;

mod utils {
    pub fn helper() {}
}
'''
        
        # C test files
        files['example.c'] = '''
#include <stdio.h>
#include <stdlib.h>

#define MAX_BUFFER 256

typedef struct {
    int id;
    char name[MAX_BUFFER];
} User;

union Data {
    int integer;
    float floating;
};

int add_numbers(int a, int b) {
    return a + b;
}

static void internal_function(void) {
    printf("Internal function\\n");
}
'''
        
        # C++ test files
        files['example.cpp'] = '''
#include <iostream>
#include <vector>
#include <memory>

namespace math {
    template<typename T>
    class Calculator {
    private:
        std::vector<T> history;
        
    public:
        Calculator() = default;
        
        T add(T a, T b) {
            T result = a + b;
            history.push_back(result);
            return result;
        }
        
        template<typename U>
        void process(U value) {
            std::cout << value << std::endl;
        }
    };
}

using namespace std;

struct Point {
    double x, y;
    Point(double x, double y) : x(x), y(y) {}
};
'''
        
        # C# test files
        files['Example.cs'] = '''
using System;
using System.Collections.Generic;
using System.Linq;

namespace MyApplication
{
    public class UserService
    {
        private readonly List<User> _users;
        
        public UserService()
        {
            _users = new List<User>();
        }
        
        public void AddUser(User user)
        {
            _users.Add(user);
        }
        
        public User FindUser(int id)
        {
            return _users.FirstOrDefault(u => u.Id == id);
        }
    }
    
    public interface IUserRepository
    {
        Task<User> GetUserAsync(int id);
    }
    
    public record UserDto(int Id, string Name);
}
'''
        
        # HTML test files
        files['index.html'] = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Test Page</title>
    <style>
        body { margin: 0; }
        .container { padding: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <h1 id="main-title">Welcome</h1>
        <p>This is a test page.</p>
        <button onclick="alert('Hello!')">Click me</button>
    </div>
    
    <script>
        function setupPage() {
            console.log('Page setup complete');
        }
        setupPage();
    </script>
</body>
</html>
'''
        
        # CSS test files
        files['styles.css'] = '''
@import url('https://fonts.googleapis.com/css2?family=Open+Sans');

:root {
    --primary-color: #007bff;
    --secondary-color: #6c757d;
}

@media (max-width: 768px) {
    .container {
        padding: 10px;
    }
}

.header {
    background-color: var(--primary-color);
    color: white;
    padding: 1rem;
}

.header h1 {
    margin: 0;
    font-size: 2rem;
}

#navigation {
    display: flex;
    justify-content: space-between;
}

@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}

.fade-in {
    animation: fadeIn 0.3s ease-in;
}
'''
        
        # Write all test files
        for filename, content in files.items():
            file_path = Path(temp_dir) / filename
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
        
        yield temp_dir, files
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.mark.asyncio
    async def test_python_entity_extraction(self, ingestion_plugin, temp_files):
        """Test Python entity extraction."""
        temp_dir, files = temp_files
        python_file = Path(temp_dir) / 'simple.py'
        
        entities = await ingestion_plugin._parse_file(python_file, 'python')
        
        # Should extract function, class, method, and imports
        assert len(entities) >= 4
        
        entity_types = [e['entity_type'] for e in entities]
        assert 'function_definition' in entity_types
        assert 'class_definition' in entity_types
        
        # Check for specific entities
        names = [e.get('name', '') for e in entities]
        assert any('hello_world' in name for name in names)
        assert any('Calculator' in name for name in names)
    
    @pytest.mark.asyncio
    async def test_javascript_entity_extraction(self, ingestion_plugin, temp_files):
        """Test JavaScript entity extraction."""
        temp_dir, files = temp_files
        js_file = Path(temp_dir) / 'app.js'
        
        entities = await ingestion_plugin._parse_file(js_file, 'javascript')
        
        assert len(entities) >= 3
        
        entity_types = [e['entity_type'] for e in entities]
        assert 'function_declaration' in entity_types or 'function_expression' in entity_types
        assert 'class_declaration' in entity_types
        
        names = [e.get('name', '') for e in entities]
        assert any('greet' in name for name in names)
        assert any('User' in name for name in names)
    
    @pytest.mark.asyncio
    async def test_typescript_entity_extraction(self, ingestion_plugin, temp_files):
        """Test TypeScript entity extraction."""
        temp_dir, files = temp_files
        ts_file = Path(temp_dir) / 'types.ts'
        
        entities = await ingestion_plugin._parse_file(ts_file, 'typescript')
        
        assert len(entities) >= 3
        
        entity_types = [e['entity_type'] for e in entities]
        assert 'interface_declaration' in entity_types
        assert 'class_declaration' in entity_types
    
    @pytest.mark.asyncio
    async def test_java_entity_extraction(self, ingestion_plugin, temp_files):
        """Test Java entity extraction."""
        temp_dir, files = temp_files
        java_file = Path(temp_dir) / 'Example.java'
        
        entities = await ingestion_plugin._parse_file(java_file, 'java')
        
        assert len(entities) >= 3
        
        entity_types = [e['entity_type'] for e in entities]
        assert 'class_declaration' in entity_types
        assert 'method_declaration' in entity_types or 'constructor_declaration' in entity_types
        assert 'interface_declaration' in entity_types
    
    @pytest.mark.asyncio
    async def test_go_entity_extraction(self, ingestion_plugin, temp_files):
        """Test Go entity extraction."""
        temp_dir, files = temp_files
        go_file = Path(temp_dir) / 'main.go'
        
        entities = await ingestion_plugin._parse_file(go_file, 'go')
        
        assert len(entities) >= 3
        
        entity_types = [e['entity_type'] for e in entities]
        assert 'function_declaration' in entity_types or 'method_declaration' in entity_types
        assert 'type_declaration' in entity_types
    
    @pytest.mark.asyncio
    async def test_rust_entity_extraction(self, ingestion_plugin, temp_files):
        """Test Rust entity extraction."""
        temp_dir, files = temp_files
        rust_file = Path(temp_dir) / 'lib.rs'
        
        entities = await ingestion_plugin._parse_file(rust_file, 'rust')
        
        assert len(entities) >= 4
        
        entity_types = [e['entity_type'] for e in entities]
        assert 'struct_item' in entity_types
        assert 'impl_item' in entity_types
        assert 'trait_item' in entity_types
    
    @pytest.mark.asyncio
    async def test_c_entity_extraction(self, ingestion_plugin, temp_files):
        """Test C entity extraction."""
        temp_dir, files = temp_files
        c_file = Path(temp_dir) / 'example.c'
        
        entities = await ingestion_plugin._parse_file(c_file, 'c')
        
        assert len(entities) >= 3
        
        entity_types = [e['entity_type'] for e in entities]
        assert 'function_definition' in entity_types
        assert 'struct_specifier' in entity_types or 'union_specifier' in entity_types
    
    @pytest.mark.asyncio
    async def test_cpp_entity_extraction(self, ingestion_plugin, temp_files):
        """Test C++ entity extraction."""
        temp_dir, files = temp_files
        cpp_file = Path(temp_dir) / 'example.cpp'
        
        entities = await ingestion_plugin._parse_file(cpp_file, 'cpp')
        
        assert len(entities) >= 3
        
        entity_types = [e['entity_type'] for e in entities]
        assert 'class_specifier' in entity_types or 'struct_specifier' in entity_types
        assert 'namespace_definition' in entity_types
    
    @pytest.mark.asyncio
    async def test_csharp_entity_extraction(self, ingestion_plugin, temp_files):
        """Test C# entity extraction."""
        temp_dir, files = temp_files
        cs_file = Path(temp_dir) / 'Example.cs'
        
        entities = await ingestion_plugin._parse_file(cs_file, 'csharp')
        
        assert len(entities) >= 3
        
        entity_types = [e['entity_type'] for e in entities]
        assert 'class_declaration' in entity_types
        assert 'interface_declaration' in entity_types
        assert 'namespace_declaration' in entity_types
    
    @pytest.mark.asyncio
    async def test_html_entity_extraction(self, ingestion_plugin, temp_files):
        """Test HTML entity extraction."""
        temp_dir, files = temp_files
        html_file = Path(temp_dir) / 'index.html'
        
        entities = await ingestion_plugin._parse_file(html_file, 'html')
        
        assert len(entities) >= 3
        
        entity_types = [e['entity_type'] for e in entities]
        assert 'element' in entity_types
        assert 'doctype' in entity_types
    
    @pytest.mark.asyncio
    async def test_css_entity_extraction(self, ingestion_plugin, temp_files):
        """Test CSS entity extraction."""
        temp_dir, files = temp_files
        css_file = Path(temp_dir) / 'styles.css'
        
        entities = await ingestion_plugin._parse_file(css_file, 'css')
        
        assert len(entities) >= 3
        
        entity_types = [e['entity_type'] for e in entities]
        assert 'rule_set' in entity_types
        assert 'at_rule' in entity_types or 'media_query' in entity_types
    
    @pytest.mark.asyncio
    async def test_malformed_file_handling(self, ingestion_plugin):
        """Test handling of malformed syntax."""
        temp_dir = tempfile.mkdtemp()
        
        # Create malformed Python file
        malformed_file = Path(temp_dir) / 'malformed.py'
        with open(malformed_file, 'w') as f:
            f.write('''
def incomplete_function(
    # Missing closing parenthesis and body
    
class IncompleteClass
    # Missing colon and body
    
if True
    print("Missing colon")
''')
        
        try:
            entities = await ingestion_plugin._parse_file(malformed_file, 'python')
            
            # Should not crash and may extract some partial entities
            assert isinstance(entities, list)
            # Some entities might be extracted despite syntax errors
            
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.mark.asyncio
    async def test_large_file_handling(self, ingestion_plugin):
        """Test handling of large files."""
        temp_dir = tempfile.mkdtemp()
        
        # Create large file (should be skipped)
        large_file = Path(temp_dir) / 'large.py'
        with open(large_file, 'w') as f:
            # Write > 1MB of content
            content = 'def function_{}():\n    pass\n\n' * 50000
            f.write(content)
        
        try:
            entities = await ingestion_plugin._parse_file(large_file, 'python')
            
            # Should return empty list for large files
            assert entities == []
            
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.mark.asyncio
    async def test_binary_file_handling(self, ingestion_plugin):
        """Test handling of binary files."""
        temp_dir = tempfile.mkdtemp()
        
        # Create binary file
        binary_file = Path(temp_dir) / 'binary.py'
        with open(binary_file, 'wb') as f:
            f.write(b'\x00\x01\x02\x03\xFF\xFE\xFD')
        
        try:
            entities = await ingestion_plugin._parse_file(binary_file, 'python')
            
            # Should return empty list for binary files
            assert entities == []
            
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.mark.asyncio
    async def test_encoding_error_handling(self, ingestion_plugin):
        """Test handling of encoding errors."""
        temp_dir = tempfile.mkdtemp()
        
        # Create file with mixed encoding
        encoding_file = Path(temp_dir) / 'encoding.py'
        with open(encoding_file, 'wb') as f:
            f.write('def test():\n    print("'.encode('utf-8'))
            f.write(b'\xff\xfe')  # Invalid UTF-8 bytes
            f.write('")\n'.encode('utf-8'))
        
        try:
            entities = await ingestion_plugin._parse_file(encoding_file, 'python')
            
            # Should handle encoding errors gracefully
            assert isinstance(entities, list)
            
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_language_detection(self, ingestion_plugin):
        """Test language detection by file extensions."""
        test_cases = [
            ('script.py', 'python'),
            ('app.js', 'javascript'),
            ('types.ts', 'typescript'),
            ('Main.java', 'java'),
            ('main.go', 'go'),
            ('lib.rs', 'rust'),
            ('program.c', 'c'),
            ('program.cpp', 'cpp'),
            ('Program.cs', 'csharp'),
            ('index.html', 'html'),
            ('styles.css', 'css'),
            ('unknown.txt', None),
        ]
        
        for filename, expected_language in test_cases:
            file_path = Path(filename)
            detected = ingestion_plugin._detect_language(file_path)
            assert detected == expected_language
    
    def test_entity_id_generation(self, ingestion_plugin, temp_files):
        """Test that entity IDs are unique and stable."""
        temp_dir, files = temp_files
        python_file = Path(temp_dir) / 'simple.py'
        
        # Parse the same file twice
        entities1 = await ingestion_plugin._parse_file(python_file, 'python')
        entities2 = await ingestion_plugin._parse_file(python_file, 'python')
        
        # Should generate same entities
        assert len(entities1) == len(entities2)
        
        # IDs should be consistent
        ids1 = [e['id'] for e in entities1]
        ids2 = [e['id'] for e in entities2]
        assert ids1 == ids2
        
        # All IDs should be unique
        assert len(set(ids1)) == len(ids1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])