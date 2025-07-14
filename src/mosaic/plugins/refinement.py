"""
RefinementPlugin for Mosaic MCP Tool

Implements FR-8 (Semantic Reranking) using cross-encoder/ms-marco-MiniLM-L-12-v2 model
deployed to Azure Machine Learning Endpoint.

This plugin addresses the "lost in the middle" problem by reranking candidate documents
using a cross-encoder model that scores query-document pairs directly.
"""

import logging
from typing import List, Dict, Any, Optional
import json
import asyncio

from semantic_kernel import KernelFunction, KernelParameterMetadata
from semantic_kernel.plugin_definition import sk_function, sk_function_context_parameter
import httpx
from azure.identity import DefaultAzureCredential

from ..config.settings import MosaicSettings
from ..models.base import Document


logger = logging.getLogger(__name__)


class RefinementPlugin:
    """
    Semantic Kernel plugin for document reranking using cross-encoder model.
    
    Implements semantic reranking to address the "lost in the middle" problem
    by scoring query-document pairs using a dedicated cross-encoder model
    deployed to Azure Machine Learning.
    """
    
    def __init__(self, settings: MosaicSettings):
        """Initialize the RefinementPlugin."""
        self.settings = settings
        self.credential = DefaultAzureCredential()
        self.http_client: Optional[httpx.AsyncClient] = None
        
    async def initialize(self) -> None:
        """Initialize the RefinementPlugin and validate ML endpoint."""
        try:
            # Initialize HTTP client for ML endpoint calls
            self.http_client = httpx.AsyncClient(timeout=30.0)
            
            # Validate ML endpoint is accessible (if configured)
            if self.settings.azure_ml_endpoint_url:
                await self._validate_ml_endpoint()
            else:
                logger.warning(
                    "Azure ML endpoint not configured - reranking will use simple scoring"
                )
            
            logger.info("RefinementPlugin initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RefinementPlugin: {e}")
            raise
    
    async def _validate_ml_endpoint(self) -> None:
        """Validate that the ML endpoint is accessible."""
        try:
            # Test endpoint health/availability
            health_url = f"{self.settings.azure_ml_endpoint_url.rstrip('/')}/health"
            
            # Get access token for managed identity
            token = await self._get_access_token()
            
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            }
            
            response = await self.http_client.get(health_url, headers=headers)
            
            if response.status_code == 200:
                logger.info("Azure ML endpoint validation successful")
            else:
                logger.warning(
                    f"ML endpoint health check returned {response.status_code}. "
                    "Reranking will fall back to simple scoring."
                )
                
        except Exception as e:
            logger.warning(f"ML endpoint validation failed: {e}. Using fallback scoring.")
    
    async def _get_access_token(self) -> str:
        """Get access token for Azure ML endpoint using managed identity."""
        token = await self.credential.get_token("https://ml.azure.com/.default")
        return token.token
    
    @sk_function(
        description="Rerank documents using cross-encoder semantic scoring",
        name="rerank"
    )
    @sk_function_context_parameter(
        name="query",
        description="Original search query",
        type_="str"
    )
    @sk_function_context_parameter(
        name="documents",
        description="List of documents to rerank",
        type_="List[Document]"
    )
    async def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        """
        Rerank documents using cross-encoder model (FR-8).
        
        Args:
            query: Original search query
            documents: List of candidate documents to rerank
            
        Returns:
            Reranked list of documents with updated scores
        """
        try:
            if not documents:
                return []
            
            # If ML endpoint is available, use cross-encoder model
            if self.settings.azure_ml_endpoint_url:
                reranked_docs = await self._rerank_with_crossencoder(query, documents)
            else:
                # Fallback to simple text-based scoring
                reranked_docs = await self._rerank_with_fallback(query, documents)
            
            # Limit to top K results
            top_k = min(self.settings.rerank_top_k, len(reranked_docs))
            result = reranked_docs[:top_k]
            
            logger.info(f"Reranked {len(documents)} documents to top {len(result)}")
            return result
            
        except Exception as e:
            logger.error(f"Document reranking failed: {e}")
            # Return original documents on error
            return documents[:self.settings.rerank_top_k]
    
    async def _rerank_with_crossencoder(
        self, 
        query: str, 
        documents: List[Document]
    ) -> List[Document]:
        """Rerank documents using Azure ML cross-encoder endpoint."""
        try:
            # Prepare query-document pairs for the model
            pairs = []
            for doc in documents:
                pairs.append({
                    "query": query,
                    "document": doc.content[:512],  # Truncate for model limits
                    "doc_id": doc.id
                })
            
            # Get access token
            token = await self._get_access_token()
            
            # Call Azure ML endpoint
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "instances": pairs,
                "parameters": {
                    "return_scores": True,
                    "batch_size": min(32, len(pairs))  # Reasonable batch size
                }
            }
            
            response = await self.http_client.post(
                f"{self.settings.azure_ml_endpoint_url}/score",
                headers=headers,
                json=payload
            )
            
            if response.status_code != 200:
                logger.error(f"ML endpoint error {response.status_code}: {response.text}")
                return await self._rerank_with_fallback(query, documents)
            
            result = response.json()
            scores = result.get("predictions", [])
            
            # Update document scores and sort
            doc_scores = {}
            for i, score in enumerate(scores):
                if i < len(documents):
                    doc_scores[documents[i].id] = float(score)
            
            # Sort documents by new scores
            reranked = sorted(
                documents,
                key=lambda doc: doc_scores.get(doc.id, 0.0),
                reverse=True
            )
            
            # Update scores in document objects
            for doc in reranked:
                if doc.id in doc_scores:
                    doc.score = doc_scores[doc.id]
                    doc.metadata["rerank_source"] = "cross_encoder"
            
            return reranked
            
        except Exception as e:
            logger.error(f"Cross-encoder reranking failed: {e}")
            return await self._rerank_with_fallback(query, documents)
    
    async def _rerank_with_fallback(
        self, 
        query: str, 
        documents: List[Document]
    ) -> List[Document]:
        """Fallback reranking using simple text matching scores."""
        try:
            query_lower = query.lower()
            query_terms = set(query_lower.split())
            
            scored_docs = []
            
            for doc in documents:
                content_lower = doc.content.lower()
                content_terms = set(content_lower.split())
                
                # Calculate simple relevance score
                # 1. Term overlap score
                overlap = len(query_terms.intersection(content_terms))
                overlap_score = overlap / len(query_terms) if query_terms else 0
                
                # 2. Substring match bonus
                substring_bonus = 0.2 if query_lower in content_lower else 0
                
                # 3. Length penalty (prefer shorter, more focused results)
                length_penalty = max(0, 1 - (len(doc.content) / 2000))  # Penalty after 2000 chars
                
                # 4. Existing score weight (if available)
                existing_score = doc.score or 0.5
                
                # Combine scores
                final_score = (
                    0.4 * overlap_score +
                    0.3 * substring_bonus +
                    0.1 * length_penalty +
                    0.2 * existing_score
                )
                
                # Update document score
                doc.score = final_score
                doc.metadata["rerank_source"] = "fallback_scoring"
                scored_docs.append(doc)
            
            # Sort by final score
            scored_docs.sort(key=lambda x: x.score or 0, reverse=True)
            
            return scored_docs
            
        except Exception as e:
            logger.error(f"Fallback reranking failed: {e}")
            # Return original documents unchanged
            return documents
    
    async def get_status(self) -> Dict[str, Any]:
        """Get plugin status information."""
        ml_available = False
        if self.settings.azure_ml_endpoint_url and self.http_client:
            try:
                # Quick health check
                await asyncio.wait_for(self._validate_ml_endpoint(), timeout=5.0)
                ml_available = True
            except:
                pass
        
        return {
            "status": "active",
            "ml_endpoint_configured": bool(self.settings.azure_ml_endpoint_url),
            "ml_endpoint_available": ml_available,
            "rerank_mode": "cross_encoder" if ml_available else "fallback",
            "max_results": self.settings.rerank_top_k
        }
    
    async def cleanup(self) -> None:
        """Cleanup plugin resources."""
        if self.http_client:
            await self.http_client.aclose()
            self.http_client = None
        
        logger.info("RefinementPlugin cleanup completed")