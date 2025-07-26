# Mosaic Code Domain Ontologies Documentation

## Overview

This document describes the OWL 2 ontologies created for the Mosaic MCP Tool's OmniRAG implementation. These ontologies provide the semantic foundation for representing code entities, their relationships, and domain-specific programming language concepts.

## Ontology Structure

### Base Namespace Strategy

- **Base IRI**: `http://mosaic.ai/ontology/`
- **Core Ontology**: `http://mosaic.ai/ontology/code-base#`
- **Python Extensions**: `http://mosaic.ai/ontology/python#`
- **Relationships**: `http://mosaic.ai/ontology/relationships#`

### Prefix Conventions

```turtle
@prefix code: <http://mosaic.ai/ontology/code-base#> .
@prefix python: <http://mosaic.ai/ontology/python#> .
@prefix rel: <http://mosaic.ai/ontology/relationships#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
```

## Ontology Files

### 1. code_base.owl - Core Code Entity Ontology

**Purpose**: Defines fundamental classes and properties for representing software code entities.

**Key Classes**:
- `CodeEntity` - Root class for all code-related entities
- `Container` - Abstract class for entities that contain other entities
- `Function` - Callable units of code
- `Class` - Object-oriented class definitions  
- `Module` - Files/namespaces containing code
- `Library` - External dependencies
- `Project` - Top-level software projects
- `Variable` - Variable declarations and usage
- `DataType` - Programming language data types

**Key Properties**:
- Data Properties: `name`, `file_path`, `line_count`, `start_line`, `end_line`, `signature`, `complexity`
- Object Properties: `containedIn`, `contains`, `hasType`
- Annotation Properties: `author`, `creation_date`, `last_modified`

### 2. python.owl - Python-Specific Extensions

**Purpose**: Extends the core ontology with Python language-specific constructs and patterns.

**Key Classes**:
- `PythonCallable` - Abstract class for Python callable entities
- `PythonFunction` - Python function definitions
- `PythonClass` - Python class definitions
- `PythonMethod` - Methods within Python classes
- `PythonModule` - Python .py files
- `PythonPackage` - Python packages containing modules
- `PythonProperty` - Python properties with getter/setter behavior
- `PythonImport` - Python import statements
- `Decorator` - Python decorators

**Key Properties**:
- `is_async` - Boolean indicating asynchronous functions
- `is_generator` - Boolean indicating generator functions
- `has_docstring` - Boolean indicating presence of docstrings
- `import_alias` - String for import aliases
- `decoratedBy` - Object property linking functions to decorators
- `hasBaseClass` - Transitive property for class inheritance

### 3. relationships.owl - Code Relationship Properties

**Purpose**: Defines object properties that express relationships between code entities.

**Key Relationships**:

1. **Function Call Relationships**:
   - `calls` / `calledBy` - Function invocation relationships

2. **Containment Relationships**:
   - `definedIn` / `defines` - Where entities are defined
   - `containedIn` / `contains` - Physical containment

3. **Dependency Relationships**:
   - `dependsOn` / `dependencyOf` - Code dependencies (transitive)
   - `uses` / `usedBy` - Library usage

4. **Object-Oriented Relationships**:
   - `inheritsFrom` / `inheritedBy` - Class inheritance (transitive)
   - `implements` / `implementedBy` - Interface implementation

5. **Module Relationships**:
   - `imports` / `importedBy` - Module import relationships

## Property Characteristics

### Functional Properties
- `definedIn` - Each entity is defined in exactly one container
- `hasType` - Each entity has at most one primary type
- `name` - Each entity has exactly one name
- `file_path` - Each entity has exactly one file location

### Transitive Properties  
- `dependsOn` - Dependencies can be chained
- `inheritsFrom` - Inheritance hierarchies
- `hasBaseClass` - Python-specific inheritance

### Inverse Properties
All major relationships have defined inverse properties to enable bi-directional traversal.

## Usage Examples

### Creating Code Entity Instances

```turtle
# Python function example
python:my_function a python:PythonFunction ;
    code:name "calculate_total" ;
    code:file_path "/src/utils.py" ;
    code:start_line 45 ;
    code:end_line 52 ;
    python:is_async false ;
    python:has_docstring true ;
    rel:definedIn python:utils_module .

# Python class example
python:my_class a python:PythonClass ;
    code:name "DataProcessor" ;
    code:file_path "/src/processor.py" ;
    rel:inheritsFrom python:base_processor ;
    rel:defines python:process_method .
```

### Expressing Relationships

```turtle
# Function calls
python:main_function rel:calls python:helper_function .

# Module imports
python:main_module rel:imports python:utils_module .

# Class inheritance
python:child_class rel:inheritsFrom python:parent_class .

# Library dependencies
python:my_project rel:uses code:flask_library .
```

## Validation and Consistency

### OWL 2 DL Compliance
All ontologies are designed to comply with OWL 2 DL profile for:
- Decidable reasoning
- Tool compatibility
- Performance optimization

### Consistency Rules
- No circular inheritance relationships
- Proper domain/range restrictions
- Functional property constraints
- Inverse property definitions

## Integration with Mosaic MCP Tool

These ontologies serve as the semantic foundation for:

1. **AST to RDF Triple Generation**: Converting parsed code into semantic triples
2. **SPARQL Query Generation**: Enabling complex graph queries
3. **Intent Detection**: Understanding query types and routing strategies  
4. **Context Aggregation**: Combining results from multiple sources
5. **Relationship Traversal**: Following connections between code entities

## Extensibility

The ontology design supports extension for:
- Additional programming languages (JavaScript, Java, C#, etc.)
- Domain-specific patterns (web frameworks, ML libraries, etc.)  
- Custom relationship types
- Tool-specific metadata
- Quality metrics and analysis results

## Version Information

- **Version**: 1.0.0
- **OWL Version**: OWL 2 DL
- **Created**: 2025-07-25
- **Task**: OMR-P1-002 (Design and Create Code Domain Ontologies)
- **Dependencies**: FOAF ontology for basic metadata

## References

- [OWL 2 Web Ontology Language Overview](https://www.w3.org/TR/owl2-overview/)
- [OWL 2 DL Profile](https://www.w3.org/TR/owl2-profiles/#OWL_2_DL)
- [Mosaic MCP Tool Architecture](https://github.com/ChrisMcKee1/Mosaic)