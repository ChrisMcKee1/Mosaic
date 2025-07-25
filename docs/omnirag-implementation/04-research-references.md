# Research References and Links

## 📚 Overview

This document contains all research materials, papers, documentation links, and code references needed for implementing the OmniRAG pattern in the Mosaic MCP Tool. Each link is categorized by implementation phase and includes specific focus areas and practice exercises.

## 🔗 Foundational Research

### Core OmniRAG Pattern
- **CosmosAIGraph Repository**: https://github.com/AzureCosmosDB/CosmosAIGraph
  - **Focus**: Complete reference implementation of OmniRAG pattern
  - **Key Files**: 
    - `/impl/web_app/src/services/ontology_service.py`
    - `/impl/web_app/src/util/owl_explorer.py`
    - `/impl/web_app/src/services/ai_service.py`
    - `/impl/web_app/src/services/graph_service.py`
  - **Practice**: Clone and analyze their architecture patterns

- **Microsoft Learn CosmosAIGraph Documentation**: https://learn.microsoft.com/en-us/azure/cosmos-db/gen-ai/cosmos-ai-graph
  - **Focus**: Official documentation and implementation guidance
  - **Key Sections**: OmniRAG pattern explanation, strategy selection examples
  - **Practice**: Understand intent detection and multi-source coordination

- **CosmosAIGraph Understanding the Code**: https://raw.githubusercontent.com/AzureCosmosDB/CosmosAIGraph/7cb669852eafabdda650c41694c48aad8380d2cb/docs/understanding_the_code.md
  - **Focus**: Detailed code structure and component relationships
  - **Practice**: Map their architecture to your two-service setup

## 📖 Phase 1: RDF Infrastructure Research

### RDF and Semantic Web Fundamentals
- **RDFLib Documentation**: https://rdflib.readthedocs.io/en/stable/
  - **Focus**: Python RDF library fundamentals
  - **Key Sections**: 
    - Getting Started: https://rdflib.readthedocs.io/en/stable/gettingstarted.html
    - Intro to Creating RDF: https://rdflib.readthedocs.io/en/stable/intro_to_creating_rdf.html
    - Intro to SPARQL: https://rdflib.readthedocs.io/en/stable/intro_to_sparql.html
  - **Practice**: Create simple RDF graphs with code entities

- **OWL 2 Web Ontology Language Primer**: https://www.w3.org/TR/owl2-primer/
  - **Focus**: OWL ontology design and best practices
  - **Key Sections**: Classes, properties, restrictions, ontology design patterns
  - **Practice**: Design a simple function ontology in Protégé

- **W3C RDF Primer**: https://www.w3.org/TR/rdf11-primer/
  - **Focus**: RDF concepts, triples, and graph modeling
  - **Key Sections**: RDF basics, vocabulary design, linking data
  - **Practice**: Model code relationships as RDF triples

### Azure Digital Twins and Ontology Conversion
- **Convert Industry Ontologies to DTDL**: https://learn.microsoft.com/en-us/azure/digital-twins/concepts-ontologies-convert
  - **Focus**: Ontology conversion patterns and tools
  - **Key Sections**: Conversion patterns, RDF to DTDL mapping
  - **Practice**: Understand how to convert between ontology formats

- **OWL2DTDL Converter Sample**: https://github.com/Azure/opendigitaltwins-tools/tree/master/OWL2DTDL
  - **Focus**: Working code for ontology conversion
  - **Practice**: Study conversion patterns for your code ontologies

### RDF Tools and Libraries
- **Apache Jena Documentation**: https://jena.apache.org/documentation/
  - **Focus**: Java RDF framework (reference for understanding patterns)
  - **Key Sections**: RDF API, SPARQL queries, reasoning
  - **Practice**: Understand RDF processing patterns

- **Protégé Ontology Editor**: https://protege.stanford.edu/
  - **Focus**: Visual ontology design and validation
  - **Practice**: Create and validate your code ontologies

## 📖 Phase 2: SPARQL Integration Research

### SPARQL Query Language
- **SPARQL 1.1 Query Language**: https://www.w3.org/TR/sparql11-query/
  - **Focus**: Complete SPARQL specification and advanced features
  - **Key Sections**: 
    - Basic Graph Patterns: https://www.w3.org/TR/sparql11-query/#BasicGraphPatterns
    - Property Paths: https://www.w3.org/TR/sparql11-query/#propertypaths
    - Aggregation: https://www.w3.org/TR/sparql11-query/#aggregates
  - **Practice**: Write complex queries for code relationship scenarios

- **SPARQL 1.1 Update**: https://www.w3.org/TR/sparql11-update/
  - **Focus**: Updating RDF graphs programmatically
  - **Practice**: Create update queries for dynamic code analysis

### Natural Language to SPARQL Translation
- **Neural Machine Translation for SPARQL**: https://arxiv.org/abs/2106.09675
  - **Focus**: State-of-the-art NL2SPARQL translation techniques
  - **Key Insights**: Template-based vs neural approaches, evaluation metrics
  - **Practice**: Implement template matching for common code queries

- **LC-QuAD 2.0 Dataset**: https://figshare.com/projects/LC-QuAD_2_0/70131
  - **Focus**: Large-scale question answering dataset for SPARQL
  - **Practice**: Study question patterns and their SPARQL translations

- **Question Answering over Knowledge Graphs**: https://arxiv.org/abs/2107.08591
  - **Focus**: Comprehensive survey of QA techniques over KGs
  - **Practice**: Choose appropriate techniques for code domain

### Azure OpenAI Integration
- **Azure OpenAI Structured Outputs**: https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/structured-outputs
  - **Focus**: Generating JSON and structured responses from Azure OpenAI
  - **Key Sections**: Pydantic models, response formatting, validation
  - **Practice**: Generate SPARQL queries with validation

- **Azure OpenAI Function Calling**: https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/function-calling
  - **Focus**: Using functions to enhance LLM capabilities
  - **Practice**: Create SPARQL generation functions

## 📖 Phase 3: OmniRAG Orchestration Research

### Intent Classification and Query Understanding
- **Intent Classification for Information Retrieval**: https://arxiv.org/abs/2010.12421
  - **Focus**: Multi-class intent classification techniques and evaluation
  - **Key Insights**: Feature engineering, confidence scoring, active learning
  - **Practice**: Build and evaluate intent classifier with your data

- **Query Intent Detection in Web Search**: https://dl.acm.org/doi/10.1145/1835449.1835643
  - **Focus**: Real-world query classification at scale
  - **Practice**: Adapt web search techniques to code search domain

- **BERT for Query Classification**: https://arxiv.org/abs/1810.04805
  - **Focus**: Using transformer models for query understanding
  - **Practice**: Fine-tune BERT for code query classification

### Multi-Source Information Fusion
- **Multi-Source Information Fusion**: https://dl.acm.org/doi/10.1145/3397271.3401075
  - **Focus**: Strategies for combining results from multiple retrieval systems
  - **Key Insights**: Confidence weighting, redundancy elimination, quality assessment
  - **Practice**: Implement weighted combination and ranked merge strategies

- **Late Fusion in Retrieval**: https://sigir.org/wp-content/uploads/2019/01/p1066.pdf
  - **Focus**: Combining ranked lists from multiple retrieval systems
  - **Practice**: Implement CombSUM, CombMNZ, and other fusion methods

- **Learning to Rank for Information Retrieval**: https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-82.pdf
  - **Focus**: Machine learning approaches to result ranking
  - **Practice**: Train ranking models for your domain

### Hybrid Graph-Vector Search
- **Hybrid Graph-Vector Search Systems**: https://arxiv.org/abs/2106.06139
  - **Focus**: Combining graph traversal with vector similarity search
  - **Key Insights**: Parallel retrieval, result aggregation, performance optimization
  - **Practice**: Implement parallel graph and vector queries

- **Graph Neural Networks for Recommendation**: https://arxiv.org/abs/2011.02260
  - **Focus**: Using graphs to enhance recommendation systems
  - **Practice**: Apply graph-based scoring to code recommendations

### Semantic Reranking and Cross-Encoders
- **Cross-Encoder vs Bi-Encoder for Ranking**: https://arxiv.org/abs/1908.10084
  - **Focus**: Comparing different neural ranking architectures
  - **Key Insights**: When to use cross-encoders vs bi-encoders, performance trade-offs
  - **Practice**: Implement cross-encoder reranking for your results

- **MS MARCO Passage Ranking**: https://microsoft.github.io/msmarco/
  - **Focus**: Large-scale passage ranking dataset and models
  - **Practice**: Use pre-trained models for semantic reranking

## 🔧 Technical Implementation References

### Azure Services Integration
- **Azure Cosmos DB Vector Search**: https://learn.microsoft.com/en-us/azure/cosmos-db/nosql/vector-search
  - **Focus**: Native vector search capabilities in Cosmos DB
  - **Practice**: Optimize vector queries for your data

- **Azure Machine Learning Deployment**: https://learn.microsoft.com/en-us/azure/machine-learning/how-to-deploy-online-endpoints
  - **Focus**: Deploying ranking models as REST endpoints
  - **Practice**: Deploy cross-encoder model for reranking

- **Azure Container Apps Environment Variables**: https://learn.microsoft.com/en-us/azure/container-apps/environment-variables
  - **Focus**: Configuration management for containerized applications
  - **Practice**: Set up environment-specific configuration

### Python Libraries and Frameworks
- **Sentence Transformers Documentation**: https://www.sbert.net/
  - **Focus**: Semantic text embeddings and similarity
  - **Key Models**: all-MiniLM-L6-v2, all-mpnet-base-v2
  - **Practice**: Choose optimal embedding models for code text

- **Scikit-learn Classification**: https://scikit-learn.org/stable/modules/classification.html
  - **Focus**: Machine learning classification algorithms
  - **Key Algorithms**: Random Forest, SVM, Logistic Regression
  - **Practice**: Compare classification algorithms for intent detection

- **PyTorch Documentation**: https://pytorch.org/docs/stable/index.html
  - **Focus**: Deep learning framework for custom models
  - **Practice**: Implement custom ranking models if needed

- **Transformers Library**: https://huggingface.co/docs/transformers/index
  - **Focus**: Pre-trained transformer models for NLP
  - **Practice**: Use BERT/RoBERTa for query understanding

## 📊 Evaluation and Benchmarking

### Information Retrieval Evaluation
- **TREC Evaluation Methodology**: https://trec.nist.gov/pubs/trec16/appendices/measures.pdf
  - **Focus**: Standard IR evaluation metrics (MAP, NDCG, MRR)
  - **Practice**: Evaluate your retrieval system performance

- **Reciprocal Rank Fusion**: https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf
  - **Focus**: Simple but effective rank fusion method
  - **Practice**: Implement RRF for combining ranked lists

### Code Search Evaluation
- **CodeSearchNet Dataset**: https://github.com/github/CodeSearchNet
  - **Focus**: Large-scale code search evaluation dataset
  - **Practice**: Adapt evaluation methods for your code domain

- **Code Search Evaluation Metrics**: https://arxiv.org/abs/1909.09436
  - **Focus**: Specialized metrics for code retrieval evaluation
  - **Practice**: Design evaluation framework for code-specific tasks

## 🔍 Monitoring and Operations

### System Monitoring
- **Azure Monitor for Container Apps**: https://learn.microsoft.com/en-us/azure/container-apps/monitor
  - **Focus**: Monitoring containerized applications in Azure
  - **Practice**: Set up comprehensive monitoring for your services

- **OpenTelemetry Python**: https://opentelemetry.io/docs/instrumentation/python/
  - **Focus**: Distributed tracing and observability
  - **Practice**: Add tracing to your OmniRAG pipeline

### Performance Optimization
- **Python Async Best Practices**: https://docs.python.org/3/library/asyncio-dev.html
  - **Focus**: Writing efficient asynchronous Python code
  - **Practice**: Optimize your parallel retrieval implementation

- **Cosmos DB Performance Best Practices**: https://learn.microsoft.com/en-us/azure/cosmos-db/nosql/performance-tips
  - **Focus**: Optimizing queries and operations in Cosmos DB
  - **Practice**: Tune your database operations for performance

## 📖 Research Papers by Topic

### Knowledge Graphs and Code Analysis
- **Code Property Graphs**: https://dl.acm.org/doi/10.1145/2786805.2786820
  - **Focus**: Graph-based code representation and analysis
  - **Practice**: Model code as property graphs

- **Software Repository Mining**: https://ieeexplore.ieee.org/document/6227210
  - **Focus**: Extracting knowledge from software repositories
  - **Practice**: Apply mining techniques to code ingestion

### Information Retrieval and Search
- **Dense Passage Retrieval**: https://arxiv.org/abs/2004.04906
  - **Focus**: Dense vector representations for passage retrieval
  - **Practice**: Compare with traditional TF-IDF approaches

- **ColBERT**: https://arxiv.org/abs/2004.12832
  - **Focus**: Efficient neural information retrieval
  - **Practice**: Consider for large-scale code search scenarios

## 🎯 Phase-Specific Research Priorities

### Before Phase 1 (Must Read)
1. RDFLib Getting Started Guide
2. OWL 2 Primer (Sections 1-3)
3. CosmosAIGraph ontology_service.py analysis
4. Basic RDF/SPARQL tutorials

### Before Phase 2 (Must Read)
1. SPARQL 1.1 Query Language (Sections 1-5)
2. Azure OpenAI Structured Outputs documentation
3. NL2SPARQL translation papers
4. CosmosAIGraph ai_service.py analysis

### Before Phase 3 (Must Read)
1. Intent Classification for IR paper
2. Multi-Source Information Fusion paper
3. Hybrid Graph-Vector Search paper
4. Cross-encoder ranking documentation

## 🔄 Continuous Learning Resources

### Blogs and Updates
- **Microsoft Research Blog**: https://www.microsoft.com/en-us/research/blog/
  - **Focus**: Latest research in AI and information retrieval
  - **Practice**: Stay updated on new techniques

- **Towards Data Science**: https://towardsdatascience.com/
  - **Focus**: Practical machine learning and data science
  - **Practice**: Find implementation tutorials and case studies

### Communities and Forums
- **Stack Overflow RDF/SPARQL**: https://stackoverflow.com/questions/tagged/sparql
  - **Focus**: Practical implementation questions and solutions
  - **Practice**: Learn from real-world implementation challenges

- **Semantic Web Community**: https://www.w3.org/community/
  - **Focus**: Latest developments in semantic web technologies
  - **Practice**: Stay current with standards and best practices

## 📝 Research Validation Checklist

### Phase 1 Readiness
- [ ] Can create RDF graphs with RDFLib
- [ ] Understand OWL class and property definitions  
- [ ] Can write basic SPARQL SELECT queries
- [ ] Familiar with ontology design patterns
- [ ] Understand CosmosAIGraph architecture

### Phase 2 Readiness
- [ ] Can write complex SPARQL queries with property paths
- [ ] Understand NL2SPARQL template-based approaches
- [ ] Familiar with Azure OpenAI structured outputs
- [ ] Can implement query execution with timeouts
- [ ] Know SPARQL result formatting patterns

### Phase 3 Readiness
- [ ] Understand intent classification techniques
- [ ] Know result fusion and ranking strategies
- [ ] Can implement parallel async operations
- [ ] Familiar with cross-encoder reranking
- [ ] Understand evaluation metrics for IR systems

## 🚨 Critical Research Notes

### When to Re-Research
- **Before starting each phase**: Re-read the specific research for that phase
- **When encountering issues**: Consult troubleshooting sections and communities
- **Before optimization**: Review performance best practices
- **When extending functionality**: Look for recent papers and techniques

### Research Documentation Practice
- **Take notes**: Document key insights and implementation decisions
- **Save examples**: Keep working code examples from research
- **Track changes**: Note when research influenced implementation decisions
- **Share learnings**: Document lessons learned for future reference

---

This research foundation ensures you have the right context and knowledge before implementing each phase of the OmniRAG transformation. Refer back to these resources whenever you need to refresh your understanding or explore advanced implementation details.