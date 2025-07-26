"""
Training data generation for query intent classification.
Creates synthetic training examples for GRAPH_RAG, VECTOR_RAG, DATABASE_RAG, and HYBRID intents.
"""

import random
from typing import List, Dict
from ..models.intent_models import QueryIntentType, QueryPattern, TrainingDataset


class IntentTrainingDataGenerator:
    """Generates training data for query intent classification."""

    def __init__(self, random_seed: int = 42):
        """Initialize the training data generator."""
        self.random_seed = random_seed
        random.seed(random_seed)
        self._init_patterns()

    def _init_patterns(self) -> None:
        """Initialize query patterns for each intent type."""

        # GRAPH_RAG patterns - relationship and traversal queries
        self.graph_patterns = [
            QueryPattern(
                template="What functions call {function_name}?",
                intent=QueryIntentType.GRAPH_RAG,
                entities=[
                    "calculate_total",
                    "process_data",
                    "validate_input",
                    "generate_report",
                    "handle_error",
                ],
                complexity="simple",
            ),
            QueryPattern(
                template="Show me the inheritance hierarchy of {class_name}",
                intent=QueryIntentType.GRAPH_RAG,
                entities=[
                    "BaseController",
                    "DataProcessor",
                    "ConfigManager",
                    "EventHandler",
                    "APIClient",
                ],
                complexity="medium",
            ),
            QueryPattern(
                template="How is {module_a} connected to {module_b}?",
                intent=QueryIntentType.GRAPH_RAG,
                entities=[
                    "auth_service",
                    "user_service",
                    "data_layer",
                    "api_gateway",
                    "cache_manager",
                ],
                complexity="medium",
            ),
            QueryPattern(
                template="What are the dependencies of {component}?",
                intent=QueryIntentType.GRAPH_RAG,
                entities=[
                    "payment_processor",
                    "email_service",
                    "logging_module",
                    "config_loader",
                    "db_connector",
                ],
                complexity="simple",
            ),
            QueryPattern(
                template="Trace the call path from {start_function} to {end_function}",
                intent=QueryIntentType.GRAPH_RAG,
                entities=[
                    "main",
                    "process_request",
                    "validate_user",
                    "save_data",
                    "send_response",
                ],
                complexity="complex",
            ),
            QueryPattern(
                template="Which classes implement {interface_name}?",
                intent=QueryIntentType.GRAPH_RAG,
                entities=[
                    "Serializable",
                    "Cacheable",
                    "Validatable",
                    "Configurable",
                    "Loggable",
                ],
                complexity="simple",
            ),
            QueryPattern(
                template="Show me all classes that extend {base_class}",
                intent=QueryIntentType.GRAPH_RAG,
                entities=[
                    "BaseModel",
                    "AbstractService",
                    "GenericHandler",
                    "CoreProcessor",
                    "PluginBase",
                ],
                complexity="simple",
            ),
            QueryPattern(
                template="What modules import {target_module}?",
                intent=QueryIntentType.GRAPH_RAG,
                entities=["utils", "config", "database", "auth", "logging"],
                complexity="simple",
            ),
            QueryPattern(
                template="Find the relationship between {entity_a} and {entity_b}",
                intent=QueryIntentType.GRAPH_RAG,
                entities=["User", "Order", "Product", "Category", "Supplier"],
                complexity="medium",
            ),
            QueryPattern(
                template="Show me the call graph for {function_name}",
                intent=QueryIntentType.GRAPH_RAG,
                entities=[
                    "initialize_system",
                    "process_payment",
                    "update_inventory",
                    "send_notification",
                    "cleanup_resources",
                ],
                complexity="complex",
            ),
        ]

        # VECTOR_RAG patterns - semantic similarity and content queries
        self.vector_patterns = [
            QueryPattern(
                template="Find code similar to this implementation: {code_description}",
                intent=QueryIntentType.VECTOR_RAG,
                entities=[
                    "data validation",
                    "error handling",
                    "caching logic",
                    "API authentication",
                    "file processing",
                ],
                complexity="medium",
            ),
            QueryPattern(
                template="What does this {code_element} do?",
                intent=QueryIntentType.VECTOR_RAG,
                entities=["function", "class", "method", "module", "variable"],
                complexity="simple",
            ),
            QueryPattern(
                template="Show examples of {design_pattern} pattern",
                intent=QueryIntentType.VECTOR_RAG,
                entities=["singleton", "factory", "observer", "strategy", "decorator"],
                complexity="medium",
            ),
            QueryPattern(
                template="Find functions with similar purpose to {description}",
                intent=QueryIntentType.VECTOR_RAG,
                entities=[
                    "user authentication",
                    "data encryption",
                    "file compression",
                    "email sending",
                    "log parsing",
                ],
                complexity="medium",
            ),
            QueryPattern(
                template="What libraries are used for {functionality}?",
                intent=QueryIntentType.VECTOR_RAG,
                entities=[
                    "machine learning",
                    "web scraping",
                    "image processing",
                    "data visualization",
                    "testing",
                ],
                complexity="simple",
            ),
            QueryPattern(
                template="Explain how {concept} is implemented",
                intent=QueryIntentType.VECTOR_RAG,
                entities=[
                    "caching",
                    "authentication",
                    "validation",
                    "serialization",
                    "pagination",
                ],
                complexity="medium",
            ),
            QueryPattern(
                template="Find similar error handling patterns",
                intent=QueryIntentType.VECTOR_RAG,
                entities=[
                    "exception handling",
                    "validation errors",
                    "network timeouts",
                    "resource cleanup",
                    "retry logic",
                ],
                complexity="medium",
            ),
            QueryPattern(
                template="What are examples of {algorithm_type} algorithms?",
                intent=QueryIntentType.VECTOR_RAG,
                entities=[
                    "sorting",
                    "searching",
                    "encryption",
                    "compression",
                    "optimization",
                ],
                complexity="simple",
            ),
            QueryPattern(
                template="Show me implementations similar to {implementation_type}",
                intent=QueryIntentType.VECTOR_RAG,
                entities=[
                    "REST API",
                    "database ORM",
                    "event handler",
                    "message queue",
                    "configuration parser",
                ],
                complexity="medium",
            ),
            QueryPattern(
                template="Find code that handles {use_case}",
                intent=QueryIntentType.VECTOR_RAG,
                entities=[
                    "file upload",
                    "user registration",
                    "payment processing",
                    "data backup",
                    "system monitoring",
                ],
                complexity="medium",
            ),
        ]

        # DATABASE_RAG patterns - structured data and aggregation queries
        self.database_patterns = [
            QueryPattern(
                template="How many functions are in {module_name}?",
                intent=QueryIntentType.DATABASE_RAG,
                entities=[
                    "auth_module",
                    "data_processor",
                    "api_handler",
                    "config_manager",
                    "utils",
                ],
                complexity="simple",
            ),
            QueryPattern(
                template="List all classes in the project",
                intent=QueryIntentType.DATABASE_RAG,
                entities=[],
                complexity="simple",
            ),
            QueryPattern(
                template="Show me the largest files by line count",
                intent=QueryIntentType.DATABASE_RAG,
                entities=[],
                complexity="simple",
            ),
            QueryPattern(
                template="What are all the import statements in {module}?",
                intent=QueryIntentType.DATABASE_RAG,
                entities=["main", "config", "utils", "models", "services"],
                complexity="simple",
            ),
            QueryPattern(
                template="Count the number of {file_type} files",
                intent=QueryIntentType.DATABASE_RAG,
                entities=[
                    "test",
                    "configuration",
                    "documentation",
                    "script",
                    "template",
                ],
                complexity="simple",
            ),
            QueryPattern(
                template="List all functions that take {parameter_count} parameters",
                intent=QueryIntentType.DATABASE_RAG,
                entities=["no", "one", "two", "three", "more than five"],
                complexity="simple",
            ),
            QueryPattern(
                template="Show me files modified in the last {time_period}",
                intent=QueryIntentType.DATABASE_RAG,
                entities=["week", "month", "day", "year", "quarter"],
                complexity="simple",
            ),
            QueryPattern(
                template="What are the most complex functions by {complexity_metric}?",
                intent=QueryIntentType.DATABASE_RAG,
                entities=[
                    "line count",
                    "cyclomatic complexity",
                    "parameter count",
                    "nesting depth",
                    "function calls",
                ],
                complexity="medium",
            ),
            QueryPattern(
                template="Count methods per class in {module}",
                intent=QueryIntentType.DATABASE_RAG,
                entities=[
                    "models",
                    "services",
                    "controllers",
                    "handlers",
                    "processors",
                ],
                complexity="simple",
            ),
            QueryPattern(
                template="List all {element_type} sorted by {sort_criteria}",
                intent=QueryIntentType.DATABASE_RAG,
                entities=["functions", "classes", "variables", "modules", "files"],
                complexity="simple",
            ),
        ]

        # HYBRID patterns - complex queries requiring multiple strategies
        self.hybrid_patterns = [
            QueryPattern(
                template="Find all classes that implement {interface} and show their relationships",
                intent=QueryIntentType.HYBRID,
                entities=[
                    "Serializable",
                    "Cacheable",
                    "Validatable",
                    "Configurable",
                    "Processable",
                ],
                complexity="complex",
            ),
            QueryPattern(
                template="What similar patterns exist and how are they connected?",
                intent=QueryIntentType.HYBRID,
                entities=[],
                complexity="complex",
            ),
            QueryPattern(
                template="Show me complex functions and their call graphs",
                intent=QueryIntentType.HYBRID,
                entities=[],
                complexity="complex",
            ),
            QueryPattern(
                template="Find error-prone code and trace its usage",
                intent=QueryIntentType.HYBRID,
                entities=[],
                complexity="complex",
            ),
            QueryPattern(
                template="Analyze the architecture of {component} and its dependencies",
                intent=QueryIntentType.HYBRID,
                entities=[
                    "authentication system",
                    "data layer",
                    "API gateway",
                    "caching layer",
                    "notification service",
                ],
                complexity="complex",
            ),
            QueryPattern(
                template="Find all {code_smell} and suggest improvements",
                intent=QueryIntentType.HYBRID,
                entities=[
                    "code duplications",
                    "long functions",
                    "large classes",
                    "complex conditionals",
                    "tight coupling",
                ],
                complexity="complex",
            ),
            QueryPattern(
                template="Show me the impact analysis of changing {component}",
                intent=QueryIntentType.HYBRID,
                entities=[
                    "database schema",
                    "API contract",
                    "configuration format",
                    "authentication method",
                    "caching strategy",
                ],
                complexity="complex",
            ),
            QueryPattern(
                template="Find performance bottlenecks and their call paths",
                intent=QueryIntentType.HYBRID,
                entities=[],
                complexity="complex",
            ),
            QueryPattern(
                template="Analyze code quality metrics and suggest refactoring for {module}",
                intent=QueryIntentType.HYBRID,
                entities=[
                    "payment_service",
                    "user_management",
                    "data_processing",
                    "report_generation",
                    "notification_system",
                ],
                complexity="complex",
            ),
            QueryPattern(
                template="Find similar implementations across modules and suggest consolidation",
                intent=QueryIntentType.HYBRID,
                entities=[],
                complexity="complex",
            ),
        ]

        self.all_patterns = {
            QueryIntentType.GRAPH_RAG: self.graph_patterns,
            QueryIntentType.VECTOR_RAG: self.vector_patterns,
            QueryIntentType.DATABASE_RAG: self.database_patterns,
            QueryIntentType.HYBRID: self.hybrid_patterns,
        }

    def generate_variations(
        self, pattern: QueryPattern, num_variations: int = 5
    ) -> List[str]:
        """Generate variations of a query pattern."""
        variations: List[str] = []

        for _ in range(num_variations):
            query = pattern.template

            # Replace entity placeholders with random entities
            import re

            placeholders = re.findall(r"\{([^}]+)\}", query)

            for placeholder in placeholders:
                if pattern.entities:
                    # Use pattern-specific entities
                    if (
                        "_" in placeholder
                    ):  # Handle compound placeholders like {module_a}, {module_b}
                        entity = random.choice(pattern.entities)
                    else:
                        entity = random.choice(pattern.entities)
                else:
                    # Use generic entities based on placeholder name
                    entity = self._get_generic_entity(placeholder)

                query = query.replace(f"{{{placeholder}}}", entity)

            # Add natural language variations
            query = self._add_language_variations(query)
            variations.append(query)

        return variations

    def _get_generic_entity(self, placeholder: str) -> str:
        """Get a generic entity for a placeholder."""
        generic_entities = {
            "function_name": [
                "process_data",
                "validate_input",
                "calculate_result",
                "handle_request",
            ],
            "class_name": [
                "DataProcessor",
                "ConfigManager",
                "EventHandler",
                "ServiceProvider",
            ],
            "module_name": ["auth", "utils", "config", "services", "models"],
            "interface_name": [
                "Serializable",
                "Cacheable",
                "Validatable",
                "Processable",
            ],
            "component": ["database", "cache", "queue", "service", "handler"],
            "code_element": ["function", "class", "method", "variable"],
            "design_pattern": ["singleton", "factory", "observer", "strategy"],
            "functionality": ["authentication", "validation", "caching", "logging"],
            "concept": ["inheritance", "polymorphism", "encapsulation", "abstraction"],
            "algorithm_type": ["sorting", "searching", "graph", "tree"],
            "implementation_type": [
                "REST API",
                "event handler",
                "data processor",
                "cache manager",
            ],
            "use_case": [
                "user login",
                "data export",
                "file upload",
                "report generation",
            ],
            "file_type": ["test", "config", "documentation", "script"],
            "parameter_count": ["no", "one", "two", "multiple"],
            "time_period": ["week", "month", "day", "year"],
            "complexity_metric": ["lines", "complexity", "parameters", "depth"],
            "element_type": ["functions", "classes", "modules", "files"],
            "sort_criteria": ["name", "size", "complexity", "date"],
            "code_smell": [
                "duplicated code",
                "long methods",
                "large classes",
                "complex logic",
            ],
        }

        return random.choice(generic_entities.get(placeholder, ["unknown"]))

    def _add_language_variations(self, query: str) -> str:
        """Add natural language variations to make queries more diverse."""

        # Question starters
        starters = ["", "Can you ", "Please ", "I need to ", "Help me ", "I want to "]
        if not any(
            query.startswith(s)
            for s in ["What", "How", "Show", "Find", "List", "Count", "Trace", "Which"]
        ):
            query = random.choice(starters) + query.lower()

        # Question endings
        endings = ["", "?", " please", " for me", " in the codebase"]
        if not query.endswith("?"):
            query += random.choice(endings)

        return query

    def generate_dataset(
        self, samples_per_class: int = 200, include_variations: bool = True
    ) -> TrainingDataset:
        """Generate a complete training dataset."""

        queries: List[str] = []
        intents: List[QueryIntentType] = []

        for intent_type, patterns in self.all_patterns.items():
            intent_queries: List[str] = []

            # Generate base examples from patterns
            for pattern in patterns:
                if include_variations:
                    variations = self.generate_variations(pattern, num_variations=10)
                    intent_queries.extend(variations)
                else:
                    # Use the base template with one example entity
                    if pattern.entities:
                        example = pattern.template.replace(
                            "{entity}", pattern.entities[0]
                        )
                        # Replace other placeholders with first entity or generic
                        import re

                        placeholders = re.findall(r"\{([^}]+)\}", example)
                        for placeholder in placeholders:
                            entity = (
                                pattern.entities[0]
                                if pattern.entities
                                else self._get_generic_entity(placeholder)
                            )
                            example = example.replace(f"{{{placeholder}}}", entity)
                        intent_queries.append(example)

            # Ensure we have enough samples per class
            while len(intent_queries) < samples_per_class:
                # Generate more variations from random patterns
                pattern = random.choice(patterns)
                variations = self.generate_variations(pattern, num_variations=5)
                intent_queries.extend(variations)

            # Take exactly the requested number of samples
            intent_queries = intent_queries[:samples_per_class]

            # Add to overall dataset
            queries.extend(intent_queries)
            intents.extend([intent_type] * len(intent_queries))

        # Shuffle the dataset
        combined = list(zip(queries, intents))
        random.shuffle(combined)
        queries, intents = zip(*combined)

        metadata = {
            "generator_version": "1.0",
            "samples_per_class": samples_per_class,
            "total_samples": len(queries),
            "random_seed": self.random_seed,
            "generation_method": (
                "pattern_based_with_variations"
                if include_variations
                else "pattern_based"
            ),
        }

        return TrainingDataset(
            queries=list(queries), intents=list(intents), metadata=metadata
        )

    def get_pattern_statistics(self) -> Dict[str, int]:
        """Get statistics about available patterns."""
        stats = {}
        for intent_type, patterns in self.all_patterns.items():
            stats[intent_type.value] = len(patterns)
        stats["total_patterns"] = sum(stats.values())
        return stats

    def export_patterns_to_file(self, file_path: str) -> None:
        """Export all patterns to a JSON file for inspection."""
        import json

        export_data = {}
        for intent_type, patterns in self.all_patterns.items():
            export_data[intent_type.value] = [
                {
                    "template": p.template,
                    "complexity": p.complexity,
                    "entities": p.entities,
                    "example_variations": self.generate_variations(p, 3),
                }
                for p in patterns
            ]

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)


def generate_training_data(
    samples_per_class: int = 200,
    random_seed: int = 42,
    export_patterns: bool = False,
    patterns_file: str = "query_patterns.json",
) -> TrainingDataset:
    """
    Convenience function to generate training data.

    Args:
        samples_per_class: Number of training samples per intent class
        random_seed: Random seed for reproducibility
        export_patterns: Whether to export patterns to a file
        patterns_file: File path for pattern export

    Returns:
        TrainingDataset with generated queries and labels
    """
    generator = IntentTrainingDataGenerator(random_seed=random_seed)

    if export_patterns:
        generator.export_patterns_to_file(patterns_file)

    dataset = generator.generate_dataset(samples_per_class=samples_per_class)

    print("Generated training dataset:")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Class distribution: {dataset.get_class_distribution()}")
    print(f"  Pattern statistics: {generator.get_pattern_statistics()}")

    return dataset
