# Product Requirements Document (PRD): Mosaic MCP Tool

| Field        | Value                                        |
|--------------|----------------------------------------------|
| Status       | In Development                               |
| Version      | 1.4                                         |
| Date         | 2025-07-14                                  |
| Author       | Chris McKee                                 |
| Stakeholders | AI Engineering, DevTools, Agent Platforms  |

## 1. Vision & Opportunity

### 1.1. Introduction

The next leap in AI application performance will not come from marginal improvements in model size, but from a radical enhancement in the quality and structure of the context provided to them. Today, developers building sophisticated AI-assisted coding tools and autonomous agents are forced to create bespoke, brittle, and inefficient systems for managing context, memory, and knowledge retrieval. This duplicates effort, introduces common failure modes, and hinders the development of truly intelligent systems.

The Mosaic is envisioned as a standardized, high-performance Model Context Protocol (MCP) Tool that provides a unified, multi-layered framework for advanced context engineering. It will serve as a centralized "brain" that AI applications can connect to, offloading the complex tasks of knowledge retrieval, dependency analysis, memory management, and context refinement. By providing this capability as a managed, pluggable service, we can dramatically accelerate development, improve AI reliability, and unlock new capabilities for both developers and agents.

### 1.2. The Opportunity

The market for AI developer tools and agentic platforms is expanding rapidly. By creating a foundational, best-in-class context engineering tool, we position ourselves as a critical infrastructure provider in this ecosystem. This tool will solve a direct and painful problem for AI developers, enabling them to build more powerful and reliable applications faster.

## 2. Target Audience & Personas

**AI Application Developer (Primary):** Developers building applications on top of LLMs. They are proficient in Python and are using frameworks like Semantic Kernel. They need to augment their AI with external knowledge (codebases, documents) and give it persistent memory without becoming experts in vector databases, graph theory, and retrieval algorithms.

**DevTools Engineer (Secondary):** Engineers building the next generation of AI-assisted IDEs and coding assistants. They need to provide their tools with deep codebase intelligence, including dependency awareness and architectural understanding.

**Agent Platform Architect (Secondary):** Architects designing platforms for hosting and running autonomous agents. They require robust, scalable, and interoperable solutions for long-term agent memory and state management.

## 3. The Problem

As detailed in the "Framework for Context Engineering" research report, developers face several critical challenges that Mosaic will solve:

- **Context is More Than a Prompt:** Simple RAG is insufficient. Developers need a sophisticated system that can retrieve from multiple sources, understand complex relationships (like code dependencies), and refine the context for relevance.

- **AI "Amnesia":** LLMs are stateless. Applications requiring memory—from remembering user preferences to tracking multi-step task progress—force developers to build complex, custom memory systems from scratch.

- **Dependency Blindness:** AI coding assistants frequently suggest code that breaks other parts of the application because they lack a holistic understanding of the codebase's dependency graph.

- **Contextual Noise:** Unfiltered, irrelevant information in the context window degrades LLM performance, leading to hallucinations and objective drift.

- **Lack of Interoperability:** Custom-built context solutions are siloed, preventing agents and tools from sharing knowledge or collaborating effectively. The MCP standard addresses this, but requires robust tools to be useful.

## 4. Goals & Objectives

- **Provide a Unified Context Service:** Create a single, accessible MCP endpoint that encapsulates a multi-layered context engineering pipeline.

- **Enable Codebase Intelligence:** Offer robust tools for analyzing and retrieving context from software projects, including dependency graphs.

- **Deliver Persistent Agent Memory:** Implement a sophisticated, multi-layered memory system that supports both short-term and long-term memory for autonomous agents.

- **Maximize Signal-to-Noise:** Integrate state-of-the-art reranking to ensure the context delivered to the LLM is maximally relevant and precise.

- **Embrace Modularity:** Build on a plugin-based architecture within Semantic Kernel to allow for easy extension and integration of new memory or retrieval techniques.

- **Ensure High Performance & Scalability:** Architect the solution on Azure to be scalable and performant, using Streamable HTTP transport for efficient, real-time communication.

## 5. Features & Requirements

Mosaic will be implemented as a Semantic Kernel MCP Tool, exposing its capabilities through a set of composable plugins.

### 5.1. Core Architecture (Based on Architecture A & B)

**FR-1: MCP Server Implementation:** The tool MUST be implemented as an MCP-compliant server. It will expose its capabilities (retrievers, memory) as tools and resources according to the MCP specification.

*Problem Solved:* Addresses the "Lack of Interoperability" by creating a standardized communication channel for AI components.

**FR-2: Semantic Kernel Integration:** The entire tool MUST be built using the Python Semantic Kernel. All core functionalities (retrieval, memory, ranking) will be implemented as Semantic Kernel Plugins.

*Problem Solved:* Promotes modularity and reusability, allowing developers to easily compose and extend context engineering capabilities.

**FR-3: Streamable HTTP Communication:** The server MUST use the Streamable HTTP transport protocol for all communication with the MCP client to enable real-time, non-blocking context streaming.

*Problem Solved:* Enables responsive, real-time AI applications by preventing blocking I/O during context retrieval and ensures compliance with the latest MCP specification (2025-03-26).

**FR-4: Azure Native Deployment:** The solution is deployed on Azure using Container Apps (Consumption Plan) with Azure Cosmos DB for NoSQL as the unified OmniRAG backend for vector search, graph data, and long-term memory, along with Azure Functions for memory consolidation.

*Problem Solved:* Ensures the system is scalable, reliable, and performant enough for production workloads while reducing cost and management overhead through a unified data backend following Microsoft's OmniRAG pattern.

### 5.2. Plugin: Multi-Pronged Retrieval (RetrievalPlugin)

This plugin will handle the first stage of the context funnel: gathering a broad set of candidate documents.

**FR-5: Hybrid Search:** The plugin orchestrates multiple retrieval methods in parallel (Vector Search and Keyword Search) using Azure Cosmos DB for NoSQL's integrated vector capabilities with the unified OmniRAG backend.

*Problem Solved:* Addresses "Context is More Than a Prompt" by moving beyond single-method retrieval to capture both semantic (intent-based) and lexical (keyword-based) relevance within a unified data platform.

**FR-6: Graph-Based Code Analysis (GraphCode tool):** Provide comprehensive codebase ingestion, parsing, and graph construction capabilities with the following sub-requirements:

**FR-6.1: Repository Ingestion:** The plugin MUST provide functions to clone, access, and traverse repositories from various sources (GitHub, GitLab, local filesystem) with support for branch selection and filtering.

**FR-6.2: Code Parsing & Analysis:** The plugin MUST implement multi-language code parsing using appropriate tools (Python AST, tree-sitter for other languages) to extract code structure, dependencies, and relationships.

**FR-6.3: Graph Construction:** The plugin MUST transform parsed code into graph entities (files, functions, classes, imports) and populate the unified Cosmos DB backend with proper embeddings and relationship mappings.

**FR-6.4: Real-time Updates:** The plugin MUST support incremental graph updates through two distinct usage patterns:

**Pattern A: Repository-Based Auto-Monitoring**
- GitHub App integration with webhook subscriptions for push/PR events
- Automated branch monitoring (main, dev, + subscribed branches)
- Self-healing branch cleanup (auto-removal of deleted branches)
- Background processing queue to avoid blocking MCP responses
- Server-side automation requiring no direct MCP client interaction

**Pattern B: Local/Manual Agent-Driven Updates**
- MCP client-callable update functions with streaming progress
- AI agent responsibility for triggering graph updates
- Dependency analysis for changed files (AST parsing + import tracking)
- Prompt engineering integration: "When you modify code, update the graph"
- MCP Streamable HTTP Mode support for long-running operations

*Technical Requirements:* Support for Server-Sent Events (SSE) for progress streaming, connection management for extended HTTP operations, backpressure handling, and recoverable partial failures.

**FR-6.5: AI Integration:** The plugin MUST provide mechanisms for AI agents to insert generated code into the graph, correlate new entities with existing ones, and analyze dependency impacts.

**FR-6.6: Query Functions:** The plugin MUST expose query functions to traverse the constructed dependency graph and retrieve code context for AI-assisted development.

**Required MCP Functions:**
- `mosaic.ingestion.ingest_repository` - Full repository ingestion
- `mosaic.ingestion.parse_codebase` - Code parsing and entity extraction
- `mosaic.ingestion.subscribe_repository_branch` - Branch subscription setup
- `mosaic.ingestion.update_graph_manual` - Manual updates with streaming
- `mosaic.ingestion.analyze_code_changes` - Dependency impact analysis
- `mosaic.ingestion.get_update_progress` - Stream-compatible status
- `mosaic.ingestion.insert_generated_code` - AI-generated code integration

*Problem Solved:* Directly mitigates "Dependency Blindness" by providing the AI with a structural understanding of the codebase, allowing it to foresee the impact of changes. Enables true AI-assisted development by maintaining a live, queryable representation of the codebase.

**FR-7: Candidate Aggregation:** The plugin MUST include a function to aggregate and de-duplicate results from the different retrieval methods into a single candidate pool.

*Problem Solved:* Manages the complexity of multiple retrieval sources, creating a clean, unified input for the subsequent refinement stage.

### 5.3. Plugin: Context Refinement (RefinementPlugin)

This plugin improves the precision of the retrieved context.

**FR-8: Semantic Reranking:** Implements a function that uses a cross-encoder model (cross-encoder/ms-marco-MiniLM-L-12-v2) deployed on Azure ML Endpoint to re-score and sort candidate documents based on their relevance to a query.

*Problem Solved:* Directly addresses "Contextual Noise" and the "lost in the middle" problem by ensuring only the most relevant information is passed to the LLM in the most optimal position.

### 5.4. Plugin: Agent Memory (MemoryPlugin)

This plugin provides a persistent, multi-layered memory system for agents, inspired by Mem0.

**FR-9: Unified Memory Interface:** The plugin MUST provide a simple, unified interface for memory operations (save_memory, retrieve_memory, clear_memory).

*Problem Solved:* Solves the developer experience challenge of "AI Amnesia" by abstracting away the complexity of the underlying multi-layered memory system.

**FR-10: Multi-Layered Storage:** The backend supports a hybrid storage model with Azure Cache for Redis (Basic C0) for short-term conversational state and Azure Cosmos DB (OmniRAG pattern) for long-term persistent memory.

*Problem Solved:* Provides a robust solution to "AI Amnesia" by modeling human memory, enabling both fast access to conversational state and persistent storage of learned knowledge.

**FR-11: LLM-Powered Consolidation:** Implements an asynchronous background function using Azure Functions (Timer Trigger) with Azure OpenAI GPT-4o (2024-11-20) to extract, consolidate, and intelligently update long-term memory from recent conversations.

*Problem Solved:* Prevents memory overload and context poisoning by ensuring the agent's long-term memory remains concise, relevant, and non-redundant.

### 5.5. Plugin: Human-AI Alignment (DiagramPlugin)

This plugin facilitates shared understanding using Mermaid diagrams.

**FR-12: Mermaid Generation:** Exposes a function that takes a natural language description and generates Mermaid diagram syntax using Azure OpenAI GPT-4o as a Semantic Function.

*Problem Solved:* Lowers the barrier to creating and maintaining architectural documentation, ensuring complex systems can be easily visualized.

**FR-13: Mermaid as Context Resource:** Allow Mermaid diagram text to be stored and retrieved as a specific resource type via the MCP interface.

*Problem Solved:* Addresses the risk of architectural misalignment and "context loss" by creating a stable, machine-readable "source of truth" that both the human and AI can reference.

**FR-14: Secure MCP Endpoint:** The server MUST be secured using the MCP Authorization specification (OAuth 2.1) with Microsoft Entra ID as the identity provider.

*Problem Solved:* Ensures production-ready security by requiring authenticated and authorized access to the server's capabilities, moving security from a future concern to a core v1.0 feature.

## 6. Out of Scope for v1.0

- **Multi-Agent Collaboration:** While the MCP architecture enables this, v1.0 will focus on single-agent/single-user context. Explicit features for memory sharing and conflict resolution between multiple agents are out of scope.

- **UI/Frontend:** This is a backend tool. No user interface will be developed as part of this PRD.

## 7. Critical Implementation Note

**Code Ingestion Gap:** The current system architecture includes robust querying capabilities but lacks the fundamental code ingestion pipeline (FR-6.1 through FR-6.5). This represents a critical dependency that must be implemented before the system can fulfill its vision of AI-assisted development. The implementation should be prioritized in the following order:

1. **Repository Access & Parsing** (FR-6.1, FR-6.2)
2. **Graph Construction** (FR-6.3)  
3. **Real-time Updates** (FR-6.4)
4. **AI Integration** (FR-6.5)

**Research Requirement:** All implementation must be validated against 2025's most current best practices using the Context7 MCP tool, web search, and fetch capabilities to ensure cutting-edge technology adoption.