# Phase 1: RDF Infrastructure Implementation

## 📋 Phase Overview

**Duration**: 4-6 weeks  
**Team Size**: 2-3 developers  
**Complexity**: High  
**Dependencies**: None (foundational phase)

This phase establishes the core RDF (Resource Description Framework) infrastructure needed for the OmniRAG pattern. By the end of this phase, the ingestion service will generate RDF triples alongside existing JSON entities.

## 🎯 Phase Objectives

- [ ] Install and configure RDF processing libraries
- [ ] Create code domain ontologies (OWL files)
- [ ] Implement AST-to-RDF triple conversion
- [ ] Build ontology management system
- [ ] Establish RDF triple storage in Cosmos DB
- [ ] Create basic SPARQL query capability
- [ ] Maintain backward compatibility with existing system

## 📚 Pre-Implementation Research (Complete Before Starting)

### Required Reading (2-3 days)
1. **RDFLib Fundamentals** (4-6 hours)
   - Link: https://rdflib.readthedocs.io/en/stable/gettingstarted.html
   - Focus: Graph creation, triple manipulation, SPARQL basics
   - Practice: Create simple RDF graphs with code entities

2. **OWL Ontology Design** (6-8 hours)
   - Link: https://www.w3.org/TR/owl2-primer/
   - Focus: Classes, properties, restrictions, ontology design patterns
   - Practice: Design a simple function ontology

3. **CosmosAIGraph RDF Implementation** (4-6 hours)
   - Links: 
     - https://github.com/AzureCosmosDB/CosmosAIGraph/blob/main/impl/web_app/src/services/ontology_service.py
     - https://github.com/AzureCosmosDB/CosmosAIGraph/blob/main/impl/web_app/src/util/owl_explorer.py
   - Focus: Ontology loading patterns, RDF graph management
   - Practice: Adapt their patterns to your architecture

4. **SPARQL Query Language** (4-6 hours)
   - Link: https://www.w3.org/TR/sparql11-query/
   - Focus: Basic SELECT queries, graph patterns, property paths
   - Practice: Write queries for code relationships

### Validation Checkpoints
- [ ] Can create RDF graphs with RDFLib
- [ ] Can write basic SPARQL queries
- [ ] Understand OWL class/property definitions
- [ ] Familiar with CosmosAIGraph patterns

## 🛠️ Implementation Steps

### Step 1: Environment Setup (Week 1, Days 1-2)

#### 1.1 Install Dependencies
```bash
cd src/mosaic-ingestion/
pip install rdflib==7.1.1
pip install SPARQLWrapper==2.0.0
pip install owlready2==0.46
pip install networkx==3.2.1

# Update requirements.txt
echo "rdflib==7.1.1" >> requirements.txt
echo "SPARQLWrapper==2.0.0" >> requirements.txt  
echo "owlready2==0.46" >> requirements.txt
echo "networkx==3.2.1" >> requirements.txt
```

#### 1.2 Create Directory Structure
```bash
mkdir -p src/mosaic-ingestion/rdf
mkdir -p src/mosaic-ingestion/ontologies
mkdir -p src/mosaic-ingestion/schemas
mkdir -p src/mosaic-ingestion/tests/rdf

touch src/mosaic-ingestion/rdf/__init__.py
touch src/mosaic-ingestion/rdf/ontology_manager.py
touch src/mosaic-ingestion/rdf/triple_generator.py
touch src/mosaic-ingestion/rdf/graph_builder.py
touch src/mosaic-ingestion/rdf/sparql_builder.py
```

#### 1.3 Environment Configuration
```bash
# Add to .env files
echo "MOSAIC_ONTOLOGY_BASE_URL=http://mosaic.ai/ontology/" >> .env
echo "MOSAIC_GRAPH_NAMESPACE=http://mosaic.ai/code#" >> .env
echo "MOSAIC_RDF_STORE_TYPE=memory" >> .env
```

### Step 2: Ontology Design and Creation (Week 1, Days 3-5)

#### 2.1 Base Code Ontology (`code_base.owl`)

**Research Before Implementation**:
- Study: https://www.w3.org/TR/owl2-syntax/
- Example: https://protege.stanford.edu/ontologies/pizza/pizza.owl

```xml
<?xml version="1.0"?>
<rdf:RDF xmlns="http://mosaic.ai/ontology/code#"
     xml:base="http://mosaic.ai/ontology/code"
     xmlns:owl="http://www.w3.org/2002/07/owl#"
     xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
     xmlns:xml="http://www.w3.org/XML/1998/namespace"
     xmlns:xsd="http://www.w3.org/2001/XMLSchema#"
     xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#">
    <owl:Ontology rdf:about="http://mosaic.ai/ontology/code"/>
    
    <!-- Classes -->
    <owl:Class rdf:about="http://mosaic.ai/ontology/code#CodeEntity">
        <rdfs:label>Code Entity</rdfs:label>
        <rdfs:comment>Base class for all code entities</rdfs:comment>
    </owl:Class>
    
    <owl:Class rdf:about="http://mosaic.ai/ontology/code#Function">
        <rdfs:subClassOf rdf:resource="http://mosaic.ai/ontology/code#CodeEntity"/>
        <rdfs:label>Function</rdfs:label>
        <rdfs:comment>A function or method in source code</rdfs:comment>
    </owl:Class>
    
    <owl:Class rdf:about="http://mosaic.ai/ontology/code#Class">
        <rdfs:subClassOf rdf:resource="http://mosaic.ai/ontology/code#CodeEntity"/>
        <rdfs:label>Class</rdfs:label>
        <rdfs:comment>A class definition in source code</rdfs:comment>
    </owl:Class>
    
    <owl:Class rdf:about="http://mosaic.ai/ontology/code#Module">
        <rdfs:subClassOf rdf:resource="http://mosaic.ai/ontology/code#CodeEntity"/>
        <rdfs:label>Module</rdfs:label>
        <rdfs:comment>A module or file containing code</rdfs:comment>
    </owl:Class>
    
    <owl:Class rdf:about="http://mosaic.ai/ontology/code#Library">
        <rdfs:subClassOf rdf:resource="http://mosaic.ai/ontology/code#CodeEntity"/>
        <rdfs:label>Library</rdfs:label>
        <rdfs:comment>An external library or package</rdfs:comment>
    </owl:Class>
    
    <!-- Object Properties -->
    <owl:ObjectProperty rdf:about="http://mosaic.ai/ontology/code#definedIn">
        <rdfs:label>defined in</rdfs:label>
        <rdfs:comment>Indicates the module where an entity is defined</rdfs:comment>
        <rdfs:domain rdf:resource="http://mosaic.ai/ontology/code#CodeEntity"/>
        <rdfs:range rdf:resource="http://mosaic.ai/ontology/code#Module"/>
    </owl:ObjectProperty>
    
    <owl:ObjectProperty rdf:about="http://mosaic.ai/ontology/code#calls">
        <rdfs:label>calls</rdfs:label>
        <rdfs:comment>Indicates that one function calls another</rdfs:comment>
        <rdfs:domain rdf:resource="http://mosaic.ai/ontology/code#Function"/>
        <rdfs:range rdf:resource="http://mosaic.ai/ontology/code#Function"/>
    </owl:ObjectProperty>
    
    <owl:ObjectProperty rdf:about="http://mosaic.ai/ontology/code#inheritsFrom">
        <rdfs:label>inherits from</rdfs:label>
        <rdfs:comment>Indicates class inheritance relationship</rdfs:comment>
        <rdfs:domain rdf:resource="http://mosaic.ai/ontology/code#Class"/>
        <rdfs:range rdf:resource="http://mosaic.ai/ontology/code#Class"/>
    </owl:ObjectProperty>
    
    <owl:ObjectProperty rdf:about="http://mosaic.ai/ontology/code#dependsOn">
        <rdfs:label>depends on</rdfs:label>
        <rdfs:comment>Indicates dependency between libraries</rdfs:comment>
        <rdfs:domain rdf:resource="http://mosaic.ai/ontology/code#Library"/>
        <rdfs:range rdf:resource="http://mosaic.ai/ontology/code#Library"/>
    </owl:ObjectProperty>
    
    <!-- Data Properties -->
    <owl:DatatypeProperty rdf:about="http://mosaic.ai/ontology/code#hasName">
        <rdfs:label>has name</rdfs:label>
        <rdfs:comment>The name of a code entity</rdfs:comment>
        <rdfs:domain rdf:resource="http://mosaic.ai/ontology/code#CodeEntity"/>
        <rdfs:range rdf:resource="http://www.w3.org/2001/XMLSchema#string"/>
    </owl:DatatypeProperty>
    
    <owl:DatatypeProperty rdf:about="http://mosaic.ai/ontology/code#hasLineNumber">
        <rdfs:label>has line number</rdfs:label>
        <rdfs:comment>The line number where an entity is defined</rdfs:comment>
        <rdfs:domain rdf:resource="http://mosaic.ai/ontology/code#CodeEntity"/>
        <rdfs:range rdf:resource="http://www.w3.org/2001/XMLSchema#integer"/>
    </owl:DatatypeProperty>
</rdf:RDF>
```

#### 2.2 Python-Specific Ontology (`python.owl`)
```xml
<?xml version="1.0"?>
<rdf:RDF xmlns="http://mosaic.ai/ontology/python#"
     xml:base="http://mosaic.ai/ontology/python"
     xmlns:code="http://mosaic.ai/ontology/code#"
     xmlns:owl="http://www.w3.org/2002/07/owl#"
     xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
     xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#">
    
    <owl:Ontology rdf:about="http://mosaic.ai/ontology/python">
        <owl:imports rdf:resource="http://mosaic.ai/ontology/code"/>
    </owl:Ontology>
    
    <!-- Python-specific classes -->
    <owl:Class rdf:about="http://mosaic.ai/ontology/python#PythonFunction">
        <rdfs:subClassOf rdf:resource="http://mosaic.ai/ontology/code#Function"/>
        <rdfs:label>Python Function</rdfs:label>
    </owl:Class>
    
    <owl:Class rdf:about="http://mosaic.ai/ontology/python#PythonClass">
        <rdfs:subClassOf rdf:resource="http://mosaic.ai/ontology/code#Class"/>
        <rdfs:label>Python Class</rdfs:label>
    </owl:Class>
    
    <owl:Class rdf:about="http://mosaic.ai/ontology/python#PythonModule">
        <rdfs:subClassOf rdf:resource="http://mosaic.ai/ontology/code#Module"/>
        <rdfs:label>Python Module</rdfs:label>
    </owl:Class>
    
    <!-- Python-specific properties -->
    <owl:DatatypeProperty rdf:about="http://mosaic.ai/ontology/python#hasDecorator">
        <rdfs:label>has decorator</rdfs:label>
        <rdfs:domain rdf:resource="http://mosaic.ai/ontology/python#PythonFunction"/>
        <rdfs:range rdf:resource="http://www.w3.org/2001/XMLSchema#string"/>
    </owl:DatatypeProperty>
    
    <owl:ObjectProperty rdf:about="http://mosaic.ai/ontology/python#hasParameter">
        <rdfs:label>has parameter</rdfs:label>
        <rdfs:domain rdf:resource="http://mosaic.ai/ontology/python#PythonFunction"/>
        <rdfs:range rdf:resource="http://mosaic.ai/ontology/python#Parameter"/>
    </owl:ObjectProperty>
</rdf:RDF>
```

### Step 3: Ontology Manager Implementation (Week 2, Days 1-3)

#### 3.1 Create `ontology_manager.py`

**Research Reference**: https://raw.githubusercontent.com/AzureCosmosDB/CosmosAIGraph/7cb669852eafabdda650c41694c48aad8380d2cb/impl/web_app/src/services/ontology_service.py

```python
import os
import asyncio
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from rdflib import Graph, Namespace, URIRef
from rdflib.namespace import RDF, RDFS, OWL
import httpx

logger = logging.getLogger(__name__)

class OntologyManager:
    """
    Manages OWL ontologies for the Mosaic code domain
    Based on CosmosAIGraph ontology service patterns
    """
    
    def __init__(self):
        self.ontologies: Dict[str, Graph] = {}
        self.base_url = os.getenv("MOSAIC_ONTOLOGY_BASE_URL", "http://mosaic.ai/ontology/")
        self.ontology_dir = Path(__file__).parent.parent / "ontologies"
        self.code_namespace = Namespace("http://mosaic.ai/ontology/code#")
        self.python_namespace = Namespace("http://mosaic.ai/ontology/python#")
        
    @classmethod
    async def create(cls) -> "OntologyManager":
        """
        Factory method for async initialization
        """
        manager = cls()
        await manager.initialize()
        return manager
        
    async def initialize(self) -> None:
        """
        Initialize ontology manager by loading all ontologies
        """
        logger.info("Initializing OntologyManager...")
        
        try:
            # Load base code ontology
            await self._load_ontology("code_base", "code_base.owl")
            
            # Load language-specific ontologies
            await self._load_ontology("python", "python.owl")
            await self._load_ontology("javascript", "javascript.owl")
            
            # Load relationship ontologies
            await self._load_ontology("dependencies", "dependencies.owl")
            
            logger.info(f"Successfully loaded {len(self.ontologies)} ontologies")
            
        except Exception as e:
            logger.error(f"Failed to initialize ontologies: {e}")
            raise
            
    async def _load_ontology(self, name: str, filename: str) -> None:
        """
        Load a specific ontology from file or HTTP
        """
        try:
            # Try loading from local filesystem first
            file_path = self.ontology_dir / filename
            
            if file_path.exists():
                graph = Graph()
                graph.parse(str(file_path), format="xml")
                self.ontologies[name] = graph
                logger.info(f"Loaded ontology '{name}' from {file_path}")
            else:
                # Try loading from HTTP (for future remote ontologies)
                await self._load_ontology_from_http(name, filename)
                
        except Exception as e:
            logger.error(f"Failed to load ontology '{name}': {e}")
            raise
            
    async def _load_ontology_from_http(self, name: str, filename: str) -> None:
        """
        Load ontology from HTTP endpoint (future enhancement)
        """
        # Placeholder for remote ontology loading
        # Implementation would use httpx to fetch remote ontologies
        logger.warning(f"HTTP ontology loading not implemented for '{name}'")
        
    def get_ontology(self, name: str) -> Optional[Graph]:
        """
        Retrieve a loaded ontology by name
        """
        return self.ontologies.get(name)
        
    def get_all_classes(self, ontology_name: str = "code_base") -> list:
        """
        Get all classes from specified ontology
        """
        ontology = self.get_ontology(ontology_name)
        if not ontology:
            return []
            
        classes = []
        for subject in ontology.subjects(RDF.type, OWL.Class):
            classes.append(subject)
        return classes
        
    def get_all_properties(self, ontology_name: str = "code_base") -> list:
        """
        Get all properties from specified ontology
        """
        ontology = self.get_ontology(ontology_name)
        if not ontology:
            return []
            
        properties = []
        for subject in ontology.subjects(RDF.type, OWL.ObjectProperty):
            properties.append(subject)
        for subject in ontology.subjects(RDF.type, OWL.DatatypeProperty):
            properties.append(subject)
        return properties
        
    def create_entity_uri(self, file_path: str, entity_name: str) -> URIRef:
        """
        Create standardized URI for code entities
        """
        # Normalize file path for URI
        normalized_path = file_path.replace("\\", "/")
        return URIRef(f"file://{normalized_path}#{entity_name}")
        
    def get_class_uri(self, class_name: str, language: str = "code") -> URIRef:
        """
        Get URI for ontology class
        """
        if language == "python":
            return self.python_namespace[class_name]
        else:
            return self.code_namespace[class_name]
            
    def validate_ontology(self, ontology_name: str) -> bool:
        """
        Validate ontology consistency and completeness
        """
        ontology = self.get_ontology(ontology_name)
        if not ontology:
            return False
            
        # Basic validation checks
        try:
            # Check for required classes
            required_classes = ["CodeEntity", "Function", "Class", "Module"]
            for class_name in required_classes:
                class_uri = self.code_namespace[class_name]
                if (class_uri, RDF.type, OWL.Class) not in ontology:
                    logger.error(f"Missing required class: {class_name}")
                    return False
                    
            # Check for required properties
            required_properties = ["definedIn", "calls", "hasName"]
            for prop_name in required_properties:
                prop_uri = self.code_namespace[prop_name]
                if not any((prop_uri, RDF.type, prop_type) in ontology 
                          for prop_type in [OWL.ObjectProperty, OWL.DatatypeProperty]):
                    logger.error(f"Missing required property: {prop_name}")
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"Ontology validation failed: {e}")
            return False

# Global instance for application use
ontology_manager: Optional[OntologyManager] = None

async def get_ontology_manager() -> OntologyManager:
    """
    Get or create global ontology manager instance
    """
    global ontology_manager
    if ontology_manager is None:
        ontology_manager = await OntologyManager.create()
    return ontology_manager
```

### Step 4: Triple Generator Implementation (Week 2, Days 4-5 & Week 3, Days 1-2)

#### 4.1 Create `triple_generator.py`

**Research Before Implementation**:
- Study: https://rdflib.readthedocs.io/en/stable/intro_to_creating_rdf.html
- Review: Your existing AST parsing in `ai_code_parser.py`

```python
import ast
import logging
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from rdflib import Graph, URIRef, Literal, BNode
from rdflib.namespace import RDF, RDFS, XSD
from .ontology_manager import get_ontology_manager

logger = logging.getLogger(__name__)

class TripleGenerator:
    """
    Converts AST entities to RDF triples using domain ontologies
    """
    
    def __init__(self):
        self.ontology_manager = None
        self.graph = Graph()
        
    async def initialize(self):
        """
        Initialize with ontology manager
        """
        self.ontology_manager = await get_ontology_manager()
        
    async def generate_triples_from_ast(self, ast_entities: List[Dict], file_path: str) -> List[Dict]:
        """
        Generate RDF triples from AST entities
        
        Args:
            ast_entities: List of entities from ai_code_parser.py
            file_path: Path to the source file
            
        Returns:
            List of RDF triple dictionaries
        """
        if not self.ontology_manager:
            await self.initialize()
            
        triples = []
        file_uri = URIRef(f"file://{file_path}")
        
        # Add file/module entity
        module_triples = self._generate_module_triples(file_uri, file_path)
        triples.extend(module_triples)
        
        # Process each AST entity
        for entity in ast_entities:
            entity_triples = await self._generate_entity_triples(entity, file_uri)
            triples.extend(entity_triples)
            
        return triples
        
    def _generate_module_triples(self, file_uri: URIRef, file_path: str) -> List[Dict]:
        """
        Generate triples for the module/file itself
        """
        triples = []
        file_name = Path(file_path).name
        
        # Module type triple
        triples.append({
            "subject": str(file_uri),
            "predicate": str(RDF.type),
            "object": str(self.ontology_manager.code_namespace.Module)
        })
        
        # Module name triple
        triples.append({
            "subject": str(file_uri),
            "predicate": str(self.ontology_manager.code_namespace.hasName),
            "object": file_name
        })
        
        # File path triple
        triples.append({
            "subject": str(file_uri),
            "predicate": str(self.ontology_manager.code_namespace.hasPath),
            "object": file_path
        })
        
        return triples
        
    async def _generate_entity_triples(self, entity: Dict, file_uri: URIRef) -> List[Dict]:
        """
        Generate triples for a specific code entity
        """
        entity_type = entity.get("entity_type")
        entity_name = entity.get("name")
        
        if not entity_name:
            return []
            
        # Create entity URI
        entity_uri = self.ontology_manager.create_entity_uri(str(file_uri).replace("file://", ""), entity_name)
        
        triples = []
        
        # Generate triples based on entity type
        if entity_type == "function":
            triples.extend(self._generate_function_triples(entity, entity_uri, file_uri))
        elif entity_type == "class":
            triples.extend(self._generate_class_triples(entity, entity_uri, file_uri))
        elif entity_type == "import":
            triples.extend(self._generate_import_triples(entity, entity_uri, file_uri))
        else:
            # Generic code entity
            triples.extend(self._generate_generic_triples(entity, entity_uri, file_uri))
            
        return triples
        
    def _generate_function_triples(self, entity: Dict, entity_uri: URIRef, file_uri: URIRef) -> List[Dict]:
        """
        Generate triples specific to function entities
        """
        triples = []
        
        # Function type
        triples.append({
            "subject": str(entity_uri),
            "predicate": str(RDF.type),
            "object": str(self.ontology_manager.code_namespace.Function)
        })
        
        # Function name
        triples.append({
            "subject": str(entity_uri),
            "predicate": str(self.ontology_manager.code_namespace.hasName),
            "object": entity.get("name", "")
        })
        
        # Defined in module
        triples.append({
            "subject": str(entity_uri),
            "predicate": str(self.ontology_manager.code_namespace.definedIn),
            "object": str(file_uri)
        })
        
        # Line number
        if "line_number" in entity:
            triples.append({
                "subject": str(entity_uri),
                "predicate": str(self.ontology_manager.code_namespace.hasLineNumber),
                "object": entity["line_number"]
            })
            
        # Function parameters
        if "parameters" in entity:
            for param in entity["parameters"]:
                param_uri = URIRef(f"{entity_uri}_param_{param}")
                triples.append({
                    "subject": str(entity_uri),
                    "predicate": str(self.ontology_manager.code_namespace.hasParameter),
                    "object": str(param_uri)
                })
                
        # Function calls (if available in entity)
        if "calls" in entity:
            for called_function in entity["calls"]:
                called_uri = self.ontology_manager.create_entity_uri(str(file_uri).replace("file://", ""), called_function)
                triples.append({
                    "subject": str(entity_uri),
                    "predicate": str(self.ontology_manager.code_namespace.calls),
                    "object": str(called_uri)
                })
                
        return triples
        
    def _generate_class_triples(self, entity: Dict, entity_uri: URIRef, file_uri: URIRef) -> List[Dict]:
        """
        Generate triples specific to class entities
        """
        triples = []
        
        # Class type
        triples.append({
            "subject": str(entity_uri),
            "predicate": str(RDF.type),
            "object": str(self.ontology_manager.code_namespace.Class)
        })
        
        # Class name
        triples.append({
            "subject": str(entity_uri),
            "predicate": str(self.ontology_manager.code_namespace.hasName),
            "object": entity.get("name", "")
        })
        
        # Defined in module
        triples.append({
            "subject": str(entity_uri),
            "predicate": str(self.ontology_manager.code_namespace.definedIn),
            "object": str(file_uri)
        })
        
        # Inheritance relationships
        if "base_classes" in entity:
            for base_class in entity["base_classes"]:
                base_uri = self.ontology_manager.create_entity_uri(str(file_uri).replace("file://", ""), base_class)
                triples.append({
                    "subject": str(entity_uri),
                    "predicate": str(self.ontology_manager.code_namespace.inheritsFrom),
                    "object": str(base_uri)
                })
                
        # Class methods
        if "methods" in entity:
            for method in entity["methods"]:
                method_uri = self.ontology_manager.create_entity_uri(str(file_uri).replace("file://", ""), f"{entity['name']}.{method}")
                triples.append({
                    "subject": str(method_uri),
                    "predicate": str(self.ontology_manager.code_namespace.memberOf),
                    "object": str(entity_uri)
                })
                
        return triples
        
    def _generate_import_triples(self, entity: Dict, entity_uri: URIRef, file_uri: URIRef) -> List[Dict]:
        """
        Generate triples specific to import statements
        """
        triples = []
        
        # Import type
        triples.append({
            "subject": str(entity_uri),
            "predicate": str(RDF.type),
            "object": str(self.ontology_manager.code_namespace.Import)
        })
        
        # Import name
        triples.append({
            "subject": str(entity_uri),
            "predicate": str(self.ontology_manager.code_namespace.hasName),
            "object": entity.get("name", "")
        })
        
        # Module imports from
        imported_module = entity.get("module", entity.get("name"))
        if imported_module:
            module_uri = URIRef(f"library://{imported_module}")
            triples.append({
                "subject": str(file_uri),
                "predicate": str(self.ontology_manager.code_namespace.imports),
                "object": str(module_uri)
            })
            
        return triples
        
    def _generate_generic_triples(self, entity: Dict, entity_uri: URIRef, file_uri: URIRef) -> List[Dict]:
        """
        Generate triples for generic code entities
        """
        triples = []
        
        # Generic code entity type
        triples.append({
            "subject": str(entity_uri),
            "predicate": str(RDF.type),
            "object": str(self.ontology_manager.code_namespace.CodeEntity)
        })
        
        # Entity name
        triples.append({
            "subject": str(entity_uri),
            "predicate": str(self.ontology_manager.code_namespace.hasName),
            "object": entity.get("name", "")
        })
        
        # Defined in module
        triples.append({
            "subject": str(entity_uri),
            "predicate": str(self.ontology_manager.code_namespace.definedIn),
            "object": str(file_uri)
        })
        
        return triples
        
    def validate_triples(self, triples: List[Dict]) -> bool:
        """
        Validate generated triples for consistency
        """
        try:
            for triple in triples:
                # Check required fields
                if not all(key in triple for key in ["subject", "predicate", "object"]):
                    logger.error(f"Invalid triple structure: {triple}")
                    return False
                    
                # Check URI format for subjects and predicates
                if not (triple["subject"].startswith("file://") or 
                       triple["subject"].startswith("library://") or
                       triple["subject"].startswith("http://")):
                    logger.error(f"Invalid subject URI: {triple['subject']}")
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"Triple validation failed: {e}")
            return False
```

### Step 5: Integration with Existing Ingestion Pipeline (Week 3, Days 3-5)

#### 5.1 Modify `ingestion.py` to Generate RDF Triples

**Current Integration Point**: Find the existing entity processing in your `ingestion.py`

```python
# Add to existing ingestion.py imports
from .rdf.triple_generator import TripleGenerator
from .rdf.ontology_manager import get_ontology_manager

class IngestionPlugin:
    def __init__(self):
        # Existing initialization
        self.triple_generator = None
        
    async def initialize(self):
        """Enhanced initialization with RDF support"""
        # Existing initialization code
        
        # Initialize RDF components
        self.triple_generator = TripleGenerator()
        await self.triple_generator.initialize()
        
    async def process_file(self, file_path: str, content: str) -> List[Dict]:
        """Enhanced file processing with RDF triple generation"""
        
        # Existing AST parsing (keep unchanged)
        ast_entities = await self.parse_ast_entities(file_path, content)
        
        # NEW: Generate RDF triples
        rdf_triples = await self.triple_generator.generate_triples_from_ast(
            ast_entities, file_path
        )
        
        # Enhance entities with RDF triples
        enhanced_entities = []
        for entity in ast_entities:
            # Find related triples for this entity
            entity_name = entity.get("name")
            related_triples = [
                triple for triple in rdf_triples 
                if entity_name in triple.get("subject", "")
            ]
            
            # Add RDF triples to entity
            enhanced_entity = {
                **entity,  # Existing entity data
                "rdf_triples": related_triples,
                "sparql_indexed": True,
                "ontology_version": "1.0"
            }
            enhanced_entities.append(enhanced_entity)
            
        return enhanced_entities
```

### Step 6: Cosmos DB Schema Updates (Week 4, Days 1-2)

#### 6.1 Update Cosmos DB Storage to Include RDF Triples

**Research Reference**: Review your existing Cosmos DB storage patterns

```python
# In your existing Cosmos DB storage code, enhance the document schema:

async def store_entity_with_triples(self, entity: Dict) -> None:
    """
    Store enhanced entity with RDF triples in Cosmos DB
    """
    document = {
        # Existing fields (unchanged)
        "id": entity.get("id"),
        "type": "code_entity", 
        "entity_type": entity.get("entity_type"),
        "name": entity.get("name"),
        "content": entity.get("content"),
        "embedding": entity.get("embedding"),
        "timestamp": entity.get("timestamp"),
        
        # NEW: RDF integration fields
        "rdf_triples": entity.get("rdf_triples", []),
        "sparql_indexed": entity.get("sparql_indexed", False),
        "ontology_version": entity.get("ontology_version", "1.0"),
        "graph_context": "repository_main_branch"
    }
    
    # Store in Cosmos DB (existing logic)
    await self.cosmos_client.create_item(document)
```

### Step 7: Basic SPARQL Query Capability (Week 4, Days 3-5)

#### 7.1 Create Basic SPARQL Query Interface

```python
# Create src/mosaic-ingestion/rdf/sparql_builder.py

from typing import List, Dict, Optional
from rdflib import Graph
from rdflib.plugins.sparql import prepareQuery
import logging

logger = logging.getLogger(__name__)

class SparqlBuilder:
    """
    Build and execute basic SPARQL queries for testing
    """
    
    def __init__(self):
        self.graph = Graph()
        
    def load_triples_from_documents(self, documents: List[Dict]) -> None:
        """
        Load RDF triples from Cosmos DB documents into in-memory graph
        """
        for doc in documents:
            triples = doc.get("rdf_triples", [])
            for triple in triples:
                try:
                    self.graph.add((
                        triple["subject"],
                        triple["predicate"], 
                        triple["object"]
                    ))
                except Exception as e:
                    logger.error(f"Failed to add triple: {e}")
                    
    def query_functions_in_module(self, module_path: str) -> List[Dict]:
        """
        Find all functions defined in a specific module
        """
        query = """
        PREFIX code: <http://mosaic.ai/ontology/code#>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        
        SELECT ?function ?name
        WHERE {
            ?function rdf:type code:Function .
            ?function code:definedIn ?module .
            ?function code:hasName ?name .
            FILTER(CONTAINS(STR(?module), "%s"))
        }
        """ % module_path
        
        results = []
        for row in self.graph.query(query):
            results.append({
                "function_uri": str(row.function),
                "name": str(row.name)
            })
        return results
        
    def query_function_calls(self, function_name: str) -> List[Dict]:
        """
        Find all functions called by a specific function
        """
        query = """
        PREFIX code: <http://mosaic.ai/ontology/code#>
        
        SELECT ?called_function ?called_name
        WHERE {
            ?function code:hasName "%s" .
            ?function code:calls ?called_function .
            ?called_function code:hasName ?called_name .
        }
        """ % function_name
        
        results = []
        for row in self.graph.query(query):
            results.append({
                "called_function": str(row.called_function),
                "name": str(row.called_name)
            })
        return results
```

## 🧪 Testing Strategy (Week 5-6)

### Test Cases to Implement

#### Unit Tests
```python
# tests/rdf/test_ontology_manager.py
import pytest
from src.mosaic_ingestion.rdf.ontology_manager import OntologyManager

@pytest.mark.asyncio
async def test_ontology_loading():
    manager = await OntologyManager.create()
    assert manager.get_ontology("code_base") is not None
    
@pytest.mark.asyncio 
async def test_entity_uri_creation():
    manager = await OntologyManager.create()
    uri = manager.create_entity_uri("/path/file.py", "test_function")
    assert str(uri) == "file:///path/file.py#test_function"

# tests/rdf/test_triple_generator.py
import pytest
from src.mosaic_ingestion.rdf.triple_generator import TripleGenerator

@pytest.mark.asyncio
async def test_function_triple_generation():
    generator = TripleGenerator()
    await generator.initialize()
    
    entity = {
        "entity_type": "function",
        "name": "test_func",
        "line_number": 10,
        "parameters": ["param1", "param2"]
    }
    
    triples = await generator._generate_entity_triples(entity, "file:///test.py")
    assert len(triples) > 0
    assert any("Function" in triple["object"] for triple in triples)
```

#### Integration Tests
```python
# tests/integration/test_rdf_pipeline.py
import pytest
from src.mosaic_ingestion.plugins.ingestion import IngestionPlugin

@pytest.mark.asyncio
async def test_full_rdf_pipeline():
    """Test complete pipeline from AST to RDF triples"""
    plugin = IngestionPlugin()
    await plugin.initialize()
    
    test_code = """
def hello_world():
    print("Hello, World!")
    
class TestClass:
    def method(self):
        pass
"""
    
    entities = await plugin.process_file("test.py", test_code)
    
    # Verify RDF triples were generated
    assert all("rdf_triples" in entity for entity in entities)
    assert all(len(entity["rdf_triples"]) > 0 for entity in entities)
```

## ✅ Phase 1 Completion Checklist

- [ ] All RDF dependencies installed and working
- [ ] Base ontologies created and validated
- [ ] OntologyManager loads ontologies successfully
- [ ] TripleGenerator converts AST entities to RDF triples
- [ ] Integration with existing ingestion pipeline completed
- [ ] Cosmos DB schema updated to store RDF triples
- [ ] Basic SPARQL queries working
- [ ] All unit tests passing
- [ ] Integration tests demonstrate full pipeline
- [ ] Backward compatibility maintained
- [ ] Documentation updated

## 🚨 Common Pitfalls and Solutions

### Pitfall 1: RDF URI Consistency
**Problem**: Inconsistent URI generation leads to broken relationships
**Solution**: Use standardized URI patterns in OntologyManager

### Pitfall 2: Ontology Validation Failures
**Problem**: OWL ontologies with syntax errors
**Solution**: Use Protégé or online validators before implementation

### Pitfall 3: Performance Issues with Large Files
**Problem**: RDF triple generation is slow for large codebases
**Solution**: Implement async processing and batch operations

### Pitfall 4: Memory Usage with RDF Graphs
**Problem**: In-memory RDF graphs consume too much memory
**Solution**: Implement graph partitioning and lazy loading

## 📋 Post-Phase 1 Cleanup

1. **Remove Debug Code**: Clean up temporary logging and debug statements
2. **Optimize Performance**: Profile and optimize triple generation
3. **Update Documentation**: Document new RDF capabilities
4. **Schema Migration**: Ensure existing data is compatible
5. **Monitoring Setup**: Add metrics for RDF processing performance

---

**Next Phase**: `phase-2-sparql-integration.md` - Implementing SPARQL query execution and natural language translation