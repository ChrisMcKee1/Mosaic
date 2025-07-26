"""
Natural Language to SPARQL Translation Service.
Uses Azure OpenAI structured outputs with Pydantic models for robust query generation.
"""

import json
import logging
from typing import List, Optional, Tuple
from dataclasses import dataclass

import openai
from azure.identity import DefaultAzureCredential
from pydantic import ValidationError

from ..config.settings import get_settings
from ..models.sparql_models import (
    NL2SPARQLRequest,
    NL2SPARQLResponse,
    SPARQLQuery,
    SPARQLTemplate,
    SPARQLPrefix,
    SPARQLVariable,
    SPARQLTriplePattern,
    SPARQLGraphPattern,
    QueryType,
    CodeEntityType,
    CodeRelationType,
)


logger = logging.getLogger(__name__)


@dataclass
class TranslationConfig:
    """Configuration for NL2SPARQL translation."""

    max_tokens: int = 2000
    temperature: float = 0.1  # Low temperature for consistent outputs
    top_p: float = 0.9
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    max_retries: int = 3
    confidence_threshold: float = 0.7


class NL2SPARQLTranslator:
    """
    Natural Language to SPARQL Query Translator.

    Uses Azure OpenAI structured outputs with Pydantic models to generate
    validated SPARQL queries from natural language descriptions.
    """

    def __init__(self):
        """Initialize the translator with Azure OpenAI client and templates."""
        self.settings = get_settings()
        self.config = TranslationConfig()
        self._setup_azure_openai()
        self._load_query_templates()

    def _setup_azure_openai(self):
        """Setup Azure OpenAI client with structured output support."""
        try:
            # Use Azure Identity for authentication
            credential = DefaultAzureCredential()

            # Initialize Azure OpenAI client
            self.client = openai.AzureOpenAI(
                api_version="2024-02-01",
                azure_endpoint=f"https://{self.settings.azure_openai_service_name}.openai.azure.com",
                azure_ad_token_provider=credential.get_token(
                    "https://cognitiveservices.azure.com/.default"
                ),
            )

            logger.info("Azure OpenAI client initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Azure OpenAI client: {e}")
            raise

    def _load_query_templates(self):
        """Load predefined SPARQL query templates for common patterns."""
        self.templates = {
            "function_calls": SPARQLTemplate(
                name="function_calls",
                description="Find function call relationships",
                pattern_keywords=[
                    "calls",
                    "invokes",
                    "function",
                    "method",
                    "procedure",
                ],
                base_query=SPARQLQuery(
                    query_type=QueryType.SELECT,
                    prefixes=[
                        SPARQLPrefix(
                            prefix="code", uri="http://mosaic.dev/ontology/code#"
                        ),
                        SPARQLPrefix(
                            prefix="rdf",
                            uri="http://www.w3.org/1999/02/22-rdf-syntax-ns#",
                        ),
                    ],
                    select_variables=[
                        SPARQLVariable(name="caller"),
                        SPARQLVariable(name="callee"),
                    ],
                    where_pattern=SPARQLGraphPattern(
                        triple_patterns=[
                            SPARQLTriplePattern(
                                subject="?caller",
                                predicate="rdf:type",
                                object="code:Function",
                            ),
                            SPARQLTriplePattern(
                                subject="?caller",
                                predicate="code:calls",
                                object="?callee",
                            ),
                            SPARQLTriplePattern(
                                subject="?callee",
                                predicate="rdf:type",
                                object="code:Function",
                            ),
                        ]
                    ),
                ),
                confidence_weight=0.9,
            ),
            "inheritance_hierarchy": SPARQLTemplate(
                name="inheritance_hierarchy",
                description="Find class inheritance relationships",
                pattern_keywords=[
                    "inherits",
                    "extends",
                    "class",
                    "inheritance",
                    "parent",
                    "child",
                ],
                base_query=SPARQLQuery(
                    query_type=QueryType.SELECT,
                    prefixes=[
                        SPARQLPrefix(
                            prefix="code", uri="http://mosaic.dev/ontology/code#"
                        ),
                        SPARQLPrefix(
                            prefix="rdf",
                            uri="http://www.w3.org/1999/02/22-rdf-syntax-ns#",
                        ),
                    ],
                    select_variables=[
                        SPARQLVariable(name="subclass"),
                        SPARQLVariable(name="superclass"),
                    ],
                    where_pattern=SPARQLGraphPattern(
                        triple_patterns=[
                            SPARQLTriplePattern(
                                subject="?subclass",
                                predicate="rdf:type",
                                object="code:Class",
                            ),
                            SPARQLTriplePattern(
                                subject="?subclass",
                                predicate="code:inherits",
                                object="?superclass",
                            ),
                            SPARQLTriplePattern(
                                subject="?superclass",
                                predicate="rdf:type",
                                object="code:Class",
                            ),
                        ]
                    ),
                ),
                confidence_weight=0.85,
            ),
            "dependency_analysis": SPARQLTemplate(
                name="dependency_analysis",
                description="Find module dependencies",
                pattern_keywords=[
                    "imports",
                    "depends",
                    "uses",
                    "module",
                    "package",
                    "dependency",
                ],
                base_query=SPARQLQuery(
                    query_type=QueryType.SELECT,
                    prefixes=[
                        SPARQLPrefix(
                            prefix="code", uri="http://mosaic.dev/ontology/code#"
                        ),
                        SPARQLPrefix(
                            prefix="rdf",
                            uri="http://www.w3.org/1999/02/22-rdf-syntax-ns#",
                        ),
                    ],
                    select_variables=[
                        SPARQLVariable(name="source"),
                        SPARQLVariable(name="target"),
                    ],
                    where_pattern=SPARQLGraphPattern(
                        triple_patterns=[
                            SPARQLTriplePattern(
                                subject="?source",
                                predicate="rdf:type",
                                object="code:Module",
                            ),
                            SPARQLTriplePattern(
                                subject="?source",
                                predicate="code:imports",
                                object="?target",
                            ),
                            SPARQLTriplePattern(
                                subject="?target",
                                predicate="rdf:type",
                                object="code:Module",
                            ),
                        ]
                    ),
                ),
                confidence_weight=0.8,
            ),
        }

        logger.info(f"Loaded {len(self.templates)} SPARQL query templates")

    async def translate_query(self, request: NL2SPARQLRequest) -> NL2SPARQLResponse:
        """
        Translate natural language query to SPARQL using Azure OpenAI structured outputs.

        Args:
            request: Natural language query request

        Returns:
            SPARQL query response with confidence scoring
        """
        try:
            logger.info(f"Translating query: {request.natural_language_query}")

            # Step 1: Template matching for initial confidence
            template_match = self._match_templates(request.natural_language_query)

            # Step 2: Use Azure OpenAI structured output for query generation
            sparql_query = await self._generate_sparql_with_openai(
                request, template_match
            )

            # Step 3: Validate query structure and ontology compliance
            validation_errors = self._validate_query(sparql_query)

            # Step 4: Calculate confidence score
            confidence_score = self._calculate_confidence(
                request, sparql_query, template_match, validation_errors
            )

            # Step 5: Generate explanation and alternatives
            explanation = self._generate_explanation(
                request, sparql_query, template_match
            )
            alternatives = self._suggest_alternatives(request, sparql_query)

            # Step 6: Detect entities and relations
            detected_entities = self._detect_entities(request.natural_language_query)
            detected_relations = self._detect_relations(request.natural_language_query)

            response = NL2SPARQLResponse(
                sparql_query=sparql_query,
                confidence_score=confidence_score,
                explanation=explanation,
                detected_entities=detected_entities,
                detected_relations=detected_relations,
                suggested_alternatives=alternatives,
                validation_errors=validation_errors,
            )

            logger.info(f"Translation completed with confidence: {confidence_score}")
            return response

        except Exception as e:
            logger.error(f"Translation failed: {e}")
            raise

    def _match_templates(
        self, query: str
    ) -> Optional[Tuple[str, SPARQLTemplate, float]]:
        """Match natural language query against predefined templates."""
        query_lower = query.lower()
        best_match = None
        best_score = 0.0

        for template_name, template in self.templates.items():
            # Calculate keyword overlap score
            keyword_matches = sum(
                1
                for keyword in template.pattern_keywords
                if keyword.lower() in query_lower
            )

            if keyword_matches > 0:
                # Normalize by number of keywords and apply template weight
                score = (
                    keyword_matches / len(template.pattern_keywords)
                ) * template.confidence_weight

                if score > best_score:
                    best_score = score
                    best_match = (template_name, template, score)

        if best_match:
            logger.info(f"Template match: {best_match[0]} (score: {best_match[2]:.2f})")

        return best_match

    async def _generate_sparql_with_openai(
        self,
        request: NL2SPARQLRequest,
        template_match: Optional[Tuple[str, SPARQLTemplate, float]],
    ) -> SPARQLQuery:
        """Generate SPARQL query using Azure OpenAI structured outputs."""

        # Prepare the function schema for structured output
        function_schema = {
            "type": "function",
            "function": {
                "name": "generate_sparql_query",
                "description": "Generate a SPARQL query from natural language",
                "parameters": SPARQLQuery.model_json_schema(),
            },
        }

        # Build the system prompt with ontology and template context
        system_prompt = self._build_system_prompt(template_match)

        # Build the user prompt
        user_prompt = self._build_user_prompt(request, template_match)

        try:
            # Call Azure OpenAI with structured output
            response = await self.client.chat.completions.create(
                model=self.settings.azure_openai_deployment_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                functions=[function_schema],
                function_call={"name": "generate_sparql_query"},
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p,
            )

            # Extract and validate the structured output
            function_call = response.choices[0].message.function_call
            if function_call and function_call.name == "generate_sparql_query":
                sparql_data = json.loads(function_call.arguments)
                return SPARQLQuery(**sparql_data)
            else:
                raise ValueError("No valid function call in OpenAI response")

        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            # Fallback to template-based generation
            if template_match:
                return self._generate_from_template(request, template_match[1])
            raise

    def _build_system_prompt(
        self, template_match: Optional[Tuple[str, SPARQLTemplate, float]]
    ) -> str:
        """Build system prompt with ontology and template context."""
        base_prompt = """You are an expert SPARQL query generator specializing in code analysis and software engineering queries.

Code Ontology:
- Classes: code:Function, code:Class, code:Module, code:Variable, code:Interface, code:Package
- Relations: code:calls, code:inherits, code:implements, code:imports, code:extends, code:uses, code:references, code:dependsOn, code:contains
- Namespace: http://mosaic.dev/ontology/code#

Guidelines:
1. Generate valid SPARQL queries that follow W3C standards
2. Use appropriate prefixes for all namespaces
3. Include type constraints (rdf:type) for entities
4. Use property paths (* + ?) for transitive relationships
5. Add LIMIT clauses for reasonable result sizes
6. Prefer SELECT queries unless specifically asked for CONSTRUCT/ASK/DESCRIBE
7. Use DISTINCT to avoid duplicates when appropriate"""

        if template_match:
            template_name, template, score = template_match
            base_prompt += f"""

Template Context:
- Matched template: {template_name} (confidence: {score:.2f})
- Template description: {template.description}
- Pattern keywords: {", ".join(template.pattern_keywords)}
- Use this template as a starting point but adapt as needed for the specific query"""

        return base_prompt

    def _build_user_prompt(
        self,
        request: NL2SPARQLRequest,
        template_match: Optional[Tuple[str, SPARQLTemplate, float]],
    ) -> str:
        """Build user prompt with query context."""
        prompt = f"Generate a SPARQL query for: {request.natural_language_query}"

        if request.context:
            prompt += f"\n\nAdditional context: {request.context}"

        if request.preferred_entities:
            entities = [e.value for e in request.preferred_entities]
            prompt += f"\n\nPreferred entities: {', '.join(entities)}"

        if request.preferred_relations:
            relations = [r.value for r in request.preferred_relations]
            prompt += f"\n\nPreferred relations: {', '.join(relations)}"

        if request.max_results:
            prompt += f"\n\nLimit results to: {request.max_results}"

        return prompt

    def _generate_from_template(
        self, request: NL2SPARQLRequest, template: SPARQLTemplate
    ) -> SPARQLQuery:
        """Generate SPARQL query from template as fallback."""
        # Simple template-based generation - could be enhanced
        query = template.base_query.copy(deep=True)

        # Apply request-specific modifications
        if request.max_results:
            query.limit = request.max_results

        return query

    def _validate_query(self, query: SPARQLQuery) -> List[str]:
        """Validate SPARQL query structure and ontology compliance."""
        errors = []

        try:
            # Validate Pydantic model
            query.dict()  # This will raise ValidationError if invalid
        except ValidationError as e:
            errors.extend([str(error) for error in e.errors])

        # Check for common SPARQL issues
        if query.query_type == QueryType.SELECT and not query.select_variables:
            errors.append("SELECT query must have at least one variable")

        if query.where_pattern.is_empty():
            errors.append("WHERE clause cannot be empty")

        # Validate ontology compliance
        for pattern in query.where_pattern.triple_patterns:
            if "code:" in pattern.predicate:
                predicate_name = pattern.predicate.split(":")[-1]
                if predicate_name not in [r.value for r in CodeRelationType]:
                    errors.append(f"Unknown code relation: {predicate_name}")

        return errors

    def _calculate_confidence(
        self,
        request: NL2SPARQLRequest,
        query: SPARQLQuery,
        template_match: Optional[Tuple[str, SPARQLTemplate, float]],
        validation_errors: List[str],
    ) -> float:
        """Calculate confidence score for the generated query."""
        confidence = 0.5  # Base confidence

        # Template match boost
        if template_match:
            confidence += template_match[2] * 0.3

        # Validation penalty
        if validation_errors:
            confidence -= len(validation_errors) * 0.1

        # Query complexity factors
        if query.where_pattern.triple_patterns:
            confidence += min(len(query.where_pattern.triple_patterns) * 0.05, 0.2)

        # Entity/relation alignment
        detected_entities = self._detect_entities(request.natural_language_query)
        detected_relations = self._detect_relations(request.natural_language_query)

        if detected_entities or detected_relations:
            confidence += 0.1

        return max(0.0, min(1.0, confidence))

    def _generate_explanation(
        self,
        request: NL2SPARQLRequest,
        query: SPARQLQuery,
        template_match: Optional[Tuple[str, SPARQLTemplate, float]],
    ) -> str:
        """Generate human-readable explanation of the SPARQL query."""
        explanation = f"Generated {query.query_type.value} query"

        if template_match:
            explanation += f" using {template_match[0]} template"

        if query.select_variables:
            vars_str = ", ".join(var.name for var in query.select_variables)
            explanation += f" to find {vars_str}"

        explanation += (
            f" with {len(query.where_pattern.triple_patterns)} triple patterns"
        )

        if query.limit:
            explanation += f", limited to {query.limit} results"

        return explanation

    def _suggest_alternatives(
        self, request: NL2SPARQLRequest, query: SPARQLQuery
    ) -> List[str]:
        """Suggest alternative query formulations."""
        alternatives = []

        # Suggest template alternatives
        for template_name, template in self.templates.items():
            for keyword in template.pattern_keywords:
                if keyword.lower() in request.natural_language_query.lower():
                    if template_name not in [
                        template_name for template_name, _, _ in [None] if None
                    ]:
                        alternatives.append(
                            f"Consider using {template.description.lower()}"
                        )
                    break

        # Suggest query modifications
        if not query.distinct:
            alternatives.append("Add DISTINCT to remove duplicate results")

        if not query.limit:
            alternatives.append("Add LIMIT clause to improve performance")

        return list(set(alternatives))[:3]  # Limit to top 3 unique suggestions

    def _detect_entities(self, query: str) -> List[CodeEntityType]:
        """Detect code entity types from natural language."""
        entities = []
        query_lower = query.lower()

        entity_patterns = {
            CodeEntityType.FUNCTION: ["function", "method", "procedure", "routine"],
            CodeEntityType.CLASS: ["class", "type", "object"],
            CodeEntityType.MODULE: ["module", "package", "namespace", "library"],
            CodeEntityType.VARIABLE: ["variable", "field", "property"],
            CodeEntityType.INTERFACE: ["interface", "contract", "protocol"],
            CodeEntityType.PACKAGE: ["package", "bundle", "component"],
        }

        for entity_type, patterns in entity_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                entities.append(entity_type)

        return entities

    def _detect_relations(self, query: str) -> List[CodeRelationType]:
        """Detect code relation types from natural language."""
        relations = []
        query_lower = query.lower()

        relation_patterns = {
            CodeRelationType.CALLS: ["calls", "invokes", "executes"],
            CodeRelationType.INHERITS: ["inherits", "extends", "derives"],
            CodeRelationType.IMPLEMENTS: ["implements", "realizes"],
            CodeRelationType.IMPORTS: ["imports", "includes", "uses"],
            CodeRelationType.USES: ["uses", "utilizes", "employs"],
            CodeRelationType.REFERENCES: ["references", "refers", "points"],
            CodeRelationType.DEPENDS_ON: ["depends", "requires", "needs"],
            CodeRelationType.CONTAINS: ["contains", "includes", "has"],
        }

        for relation_type, patterns in relation_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                relations.append(relation_type)

        return relations
