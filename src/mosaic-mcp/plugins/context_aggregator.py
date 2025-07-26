"""
Advanced Context Aggregation and Fusion System for OmniRAG.

This module implements sophisticated multi-source result fusion, deduplication,
and relevance scoring to provide optimal context for LLM responses.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple

try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from ..models.aggregation_models import (
    AggregatedItem,
    AggregationConfig,
    AggregationRequest,
    AggregationResult,
    AggregationStats,
    AggregationStrategy,
    ContextItem,
    DuplicateInfo,
    MultiSourceResult,
    RelevanceScore,
    SourceType,
)

logger = logging.getLogger(__name__)


class ContextAggregator:
    """
    Advanced context aggregation and fusion system.

    Coordinates results from multiple retrieval sources, performs deduplication,
    relevance scoring, and diversity optimization to provide optimal context.
    """

    def __init__(
        self, embedding_model_name: str = "all-MiniLM-L6-v2", enable_gpu: bool = True
    ):
        """
        Initialize the ContextAggregator.

        Args:
            embedding_model_name: Name of the sentence transformer model
            enable_gpu: Whether to use GPU acceleration if available
        """
        self.embedding_model_name = embedding_model_name
        self.enable_gpu = enable_gpu
        self._embedding_model: Optional[SentenceTransformer] = None
        self._stats = AggregationStats()

        # Initialize embedding model if available
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self._embedding_model = SentenceTransformer(embedding_model_name)
                logger.info(f"Initialized embedding model: {embedding_model_name}")
            except Exception as e:
                logger.warning(f"Failed to load embedding model: {e}")
                self._embedding_model = None
        else:
            logger.warning(
                "sentence-transformers not available, using fallback deduplication"
            )

    async def aggregate_results(self, request: AggregationRequest) -> AggregationResult:
        """
        Aggregate results from multiple sources with deduplication and ranking.

        Args:
            request: Aggregation request with multi-source results and config

        Returns:
            Aggregated and ranked context results
        """
        start_time = time.time()

        try:
            # Extract all context items from all sources
            all_items = self._extract_all_items(request.multi_source_result)

            if not all_items:
                return AggregationResult(
                    query=request.multi_source_result.query,
                    strategy_used=request.strategy,
                    aggregated_items=[],
                    total_items_processed=0,
                    duplicates_removed=0,
                    diversity_score=0.0,
                    aggregation_time_ms=0.0,
                )

            # Ensure embeddings are computed for similarity calculations
            await self._ensure_embeddings(all_items, request.query_embedding)

            # Perform deduplication
            deduplicated_items, duplicates_info = await self._deduplicate_items(
                all_items, request.config.similarity_threshold
            )

            # Calculate relevance scores
            scored_items = await self._calculate_relevance_scores(
                deduplicated_items,
                request.multi_source_result.query,
                request.query_embedding,
                request.config,
            )

            # Apply aggregation strategy
            final_items = await self._apply_aggregation_strategy(
                scored_items, request.strategy, request.config
            )

            # Calculate diversity score
            diversity_score = await self._calculate_diversity_score(final_items)

            # Create final result
            aggregation_time_ms = (time.time() - start_time) * 1000

            result = AggregationResult(
                query=request.multi_source_result.query,
                strategy_used=request.strategy,
                aggregated_items=final_items,
                total_items_processed=len(all_items),
                duplicates_removed=len(
                    [item for dup in duplicates_info for item in dup.duplicate_indices]
                ),
                diversity_score=diversity_score,
                aggregation_time_ms=aggregation_time_ms,
                metadata={
                    "duplicates_info": [dup.dict() for dup in duplicates_info],
                    "source_count": len(request.multi_source_result.source_results),
                },
            )

            # Update statistics
            await self._update_stats(result, request.strategy)

            return result

        except Exception as e:
            logger.error(f"Error in aggregate_results: {e}")
            return AggregationResult(
                query=request.multi_source_result.query,
                strategy_used=request.strategy,
                aggregated_items=[],
                total_items_processed=0,
                duplicates_removed=0,
                diversity_score=0.0,
                aggregation_time_ms=(time.time() - start_time) * 1000,
                metadata={"error": str(e)},
            )

    async def rank_context(
        self,
        items: List[ContextItem],
        query: str,
        config: AggregationConfig,
        query_embedding: Optional[List[float]] = None,
    ) -> List[AggregatedItem]:
        """
        Rank context items by relevance and diversity.

        Args:
            items: Context items to rank
            query: Original query text
            config: Aggregation configuration
            query_embedding: Pre-computed query embedding

        Returns:
            Ranked and scored aggregated items
        """
        if not items:
            return []

        # Ensure embeddings
        await self._ensure_embeddings(items, query_embedding)

        # Calculate relevance scores and create aggregated items
        aggregated_items = await self._calculate_relevance_scores(
            items, query, query_embedding, config
        )

        # Sort by relevance score
        aggregated_items.sort(key=lambda x: x.relevance_score.final_score, reverse=True)

        # Apply diversity optimization if enabled
        if config.diversity_weight > 0:
            aggregated_items = await self._optimize_diversity(aggregated_items, config)

        # Apply ranking to final items
        for rank, item in enumerate(aggregated_items[: config.max_items]):
            item.rank = rank + 1

        return aggregated_items[: config.max_items]

    def _extract_all_items(
        self, multi_source_result: MultiSourceResult
    ) -> List[ContextItem]:
        """Extract all context items from multiple source results."""
        all_items = []

        for source_result in multi_source_result.source_results:
            if source_result.error_message:
                logger.warning(
                    f"Source {source_result.source_id} had error: {source_result.error_message}"
                )
                continue

            for item in source_result.items:
                # Add source reliability to metadata
                item.metadata["source_reliability"] = source_result.source_reliability
                item.metadata["source_execution_time_ms"] = (
                    source_result.execution_time_ms
                )
                all_items.append(item)

        return all_items

    async def _ensure_embeddings(
        self, items: List[ContextItem], query_embedding: Optional[List[float]] = None
    ) -> None:
        """Ensure all items have embeddings computed."""
        if not self._embedding_model:
            return

        # Find items without embeddings
        items_to_embed = [item for item in items if item.embedding is None]

        if not items_to_embed:
            return

        try:
            # Extract content for embedding
            content_list = [item.content for item in items_to_embed]

            # Compute embeddings in batch
            embeddings = self._embedding_model.encode(
                content_list,
                convert_to_tensor=False,
                convert_to_numpy=True,
                batch_size=32,
            )

            # Assign embeddings back to items
            for item, embedding in zip(items_to_embed, embeddings):
                item.embedding = embedding.tolist()

        except Exception as e:
            logger.error(f"Error computing embeddings: {e}")

    async def _deduplicate_items(
        self, items: List[ContextItem], similarity_threshold: float
    ) -> Tuple[List[ContextItem], List[DuplicateInfo]]:
        """
        Remove duplicate items based on content similarity.

        Returns:
            Tuple of (deduplicated_items, duplicate_info_list)
        """
        if not items:
            return [], []

        deduplicated_items = []
        duplicates_info = []
        processed_indices = set()

        for i, item in enumerate(items):
            if i in processed_indices:
                continue

            # Find duplicates for this item
            duplicate_indices = []
            similarity_scores = []

            for j, other_item in enumerate(items[i + 1 :], start=i + 1):
                if j in processed_indices:
                    continue

                # Check for exact content match first
                if item.content.strip() == other_item.content.strip():
                    duplicate_indices.append(j)
                    similarity_scores.append(1.0)
                    processed_indices.add(j)
                    continue

                # Check semantic similarity if embeddings available
                if (
                    item.embedding
                    and other_item.embedding
                    and self._embedding_model is not None
                ):
                    similarity = self._calculate_cosine_similarity(
                        item.embedding, other_item.embedding
                    )

                    if similarity >= similarity_threshold:
                        duplicate_indices.append(j)
                        similarity_scores.append(similarity)
                        processed_indices.add(j)

            # Add primary item to deduplicated list
            deduplicated_items.append(item)

            # Record duplicate information if duplicates found
            if duplicate_indices:
                duplicates_info.append(
                    DuplicateInfo(
                        primary_item_index=i,
                        duplicate_indices=duplicate_indices,
                        similarity_scores=similarity_scores,
                        deduplication_method=(
                            "cosine_similarity" if item.embedding else "exact_match"
                        ),
                    )
                )

        return deduplicated_items, duplicates_info

    def _calculate_cosine_similarity(
        self, embedding1: List[float], embedding2: List[float]
    ) -> float:
        """Calculate cosine similarity between two embeddings."""
        try:
            import numpy as np

            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)

            # Calculate cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            return float(dot_product / (norm1 * norm2))

        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {e}")
            return 0.0

    async def _calculate_relevance_scores(
        self,
        items: List[ContextItem],
        query: str,
        query_embedding: Optional[List[float]],
        config: AggregationConfig,
    ) -> List[AggregatedItem]:
        """Calculate relevance scores for all items and convert to AggregatedItem objects."""
        from ..models.aggregation_models import AggregatedItem

        aggregated_items = []

        for item in items:
            # Calculate query similarity
            query_similarity = 0.0
            if query_embedding and item.embedding:
                query_similarity = self._calculate_cosine_similarity(
                    query_embedding, item.embedding
                )
            elif not query_embedding and item.embedding and self._embedding_model:
                # Compute query embedding on demand
                try:
                    query_emb = self._embedding_model.encode(
                        [query], convert_to_numpy=True
                    )[0]
                    query_similarity = self._calculate_cosine_similarity(
                        query_emb.tolist(), item.embedding
                    )
                except Exception as e:
                    logger.error(f"Error computing query similarity: {e}")

            # Extract other scores
            source_confidence = item.confidence_score
            authority_score = item.metadata.get("source_reliability", 0.5)
            recency_score = self._calculate_recency_score(item)

            # Calculate weighted final score
            final_score = (
                config.relevance_weight * query_similarity
                + config.authority_weight * authority_score
                + config.recency_weight * recency_score
                + 0.1 * source_confidence  # Small boost for confident sources
            )

            # Create relevance score object
            relevance_score = RelevanceScore(
                query_similarity=query_similarity,
                source_confidence=source_confidence,
                authority_score=authority_score,
                recency_score=recency_score,
                diversity_penalty=0.0,  # Will be calculated during diversity optimization
                final_score=min(final_score, 1.0),  # Cap at 1.0
            )

            # Create AggregatedItem
            aggregated_item = AggregatedItem(
                content=item.content,
                source_info=[f"{item.source_type.value}:{item.source_id}"],
                relevance_score=relevance_score,
                rank=0,  # Will be set later during ranking
                is_diverse=True,  # Will be updated during diversity optimization
                metadata=item.metadata.copy(),
            )

            aggregated_items.append(aggregated_item)

        return aggregated_items

    def _calculate_recency_score(self, item: ContextItem) -> float:
        """Calculate recency score based on item timestamp."""
        try:
            from datetime import datetime, timezone

            current_time = datetime.now(timezone.utc)
            time_diff = (current_time - item.timestamp).total_seconds()

            # Exponential decay with 1-day half-life
            half_life_seconds = 24 * 60 * 60  # 1 day
            recency_score = 2 ** (-time_diff / half_life_seconds)

            return min(recency_score, 1.0)

        except Exception as e:
            logger.error(f"Error calculating recency score: {e}")
            return 0.5  # Default moderate recency

    async def _apply_aggregation_strategy(
        self,
        items: List[ContextItem],
        strategy: AggregationStrategy,
        config: AggregationConfig,
    ) -> List[AggregatedItem]:
        """Apply the specified aggregation strategy."""
        if strategy == AggregationStrategy.WEIGHTED_FUSION:
            return await self._weighted_fusion_strategy(items, config)
        elif strategy == AggregationStrategy.RANKED_MERGE:
            return await self._ranked_merge_strategy(items, config)
        elif strategy == AggregationStrategy.DIVERSITY_FIRST:
            return await self._diversity_first_strategy(items, config)
        elif strategy == AggregationStrategy.RELEVANCE_FIRST:
            return await self._relevance_first_strategy(items, config)
        elif strategy == AggregationStrategy.BALANCED:
            return await self._balanced_strategy(items, config)
        else:
            # Default to balanced strategy
            return await self._balanced_strategy(items, config)

    async def _balanced_strategy(
        self, items: List[ContextItem], config: AggregationConfig
    ) -> List[AggregatedItem]:
        """Balanced strategy combining relevance and diversity."""
        # Sort by relevance first
        sorted_items = sorted(
            items, key=lambda x: x.relevance_score.final_score, reverse=True
        )

        # Apply diversity optimization
        optimized_items = await self._optimize_diversity(sorted_items, config)

        # Convert to aggregated items
        aggregated_items = []
        for rank, item in enumerate(optimized_items[: config.max_items]):
            aggregated_items.append(
                AggregatedItem(
                    content=item.content,
                    source_info=[f"{item.source_type}:{item.source_id}"],
                    relevance_score=item.relevance_score,
                    rank=rank + 1,
                    is_diverse=True,
                    metadata=item.metadata,
                )
            )

        return aggregated_items

    async def _optimize_diversity(
        self, items: List[AggregatedItem], config: AggregationConfig
    ) -> List[AggregatedItem]:
        """Optimize for diversity while maintaining relevance."""
        if not items or config.diversity_weight == 0:
            return items

        selected_items = []
        remaining_items = items.copy()

        # Always select the top relevance item first
        if remaining_items:
            top_item = remaining_items.pop(0)
            selected_items.append(top_item)

        # Iteratively select items that balance relevance and diversity
        while remaining_items and len(selected_items) < config.max_items:
            best_item = None
            best_score = -1.0
            best_index = -1

            for i, candidate in enumerate(remaining_items):
                # Calculate diversity penalty
                diversity_penalty = 0.0
                if candidate.embedding:
                    max_similarity = 0.0
                    for selected in selected_items:
                        if selected.embedding:
                            similarity = self._calculate_cosine_similarity(
                                candidate.embedding, selected.embedding
                            )
                            max_similarity = max(max_similarity, similarity)
                    diversity_penalty = max_similarity

                # Adjust final score with diversity penalty
                adjusted_score = (
                    candidate.relevance_score.final_score
                    - config.diversity_weight * diversity_penalty
                )

                if adjusted_score > best_score:
                    best_score = adjusted_score
                    best_item = candidate
                    best_index = i

            if best_item:
                # Update diversity penalty in relevance score
                best_item.relevance_score.diversity_penalty = (
                    config.diversity_weight
                    * (best_item.relevance_score.final_score - best_score)
                )
                best_item.relevance_score.final_score = best_score

                selected_items.append(best_item)
                remaining_items.pop(best_index)
            else:
                break

        return selected_items

    async def _weighted_fusion_strategy(
        self, items: List[ContextItem], config: AggregationConfig
    ) -> List[AggregatedItem]:
        """Weighted fusion strategy based on source reliability."""
        # Group items by source type
        source_groups: Dict[SourceType, List[ContextItem]] = {}
        for item in items:
            if item.source_type not in source_groups:
                source_groups[item.source_type] = []
            source_groups[item.source_type].append(item)

        # Weight items based on source type and reliability
        weighted_items = []
        for source_type, source_items in source_groups.items():
            # Apply source type weights
            type_weight = self._get_source_type_weight(source_type)

            for item in source_items:
                source_reliability = item.metadata.get("source_reliability", 1.0)
                weighted_score = (
                    item.relevance_score.final_score * type_weight * source_reliability
                )

                # Update final score
                item.relevance_score.final_score = min(weighted_score, 1.0)
                weighted_items.append(item)

        # Sort by weighted scores and apply diversity
        weighted_items.sort(key=lambda x: x.relevance_score.final_score, reverse=True)
        optimized_items = await self._optimize_diversity(weighted_items, config)

        return [
            AggregatedItem(
                content=item.content,
                source_info=[f"{item.source_type}:{item.source_id}"],
                relevance_score=item.relevance_score,
                rank=rank + 1,
                is_diverse=True,
                metadata=item.metadata,
            )
            for rank, item in enumerate(optimized_items[: config.max_items])
        ]

    async def _ranked_merge_strategy(
        self, items: List[ContextItem], config: AggregationConfig
    ) -> List[AggregatedItem]:
        """Ranked merge strategy maintaining source rankings."""
        # Sort by relevance and take top items
        sorted_items = sorted(
            items, key=lambda x: x.relevance_score.final_score, reverse=True
        )

        return [
            AggregatedItem(
                content=item.content,
                source_info=[f"{item.source_type}:{item.source_id}"],
                relevance_score=item.relevance_score,
                rank=rank + 1,
                is_diverse=False,  # Not optimized for diversity
                metadata=item.metadata,
            )
            for rank, item in enumerate(sorted_items[: config.max_items])
        ]

    async def _diversity_first_strategy(
        self, items: List[ContextItem], config: AggregationConfig
    ) -> List[AggregatedItem]:
        """Diversity-first strategy prioritizing unique content."""
        # Increase diversity weight for this strategy
        diversity_config = config.model_copy()
        diversity_config.diversity_weight = min(config.diversity_weight + 0.3, 0.8)

        # Sort by relevance first
        sorted_items = sorted(
            items, key=lambda x: x.relevance_score.final_score, reverse=True
        )

        # Apply strong diversity optimization
        optimized_items = await self._optimize_diversity(sorted_items, diversity_config)

        return [
            AggregatedItem(
                content=item.content,
                source_info=[f"{item.source_type}:{item.source_id}"],
                relevance_score=item.relevance_score,
                rank=rank + 1,
                is_diverse=True,
                metadata=item.metadata,
            )
            for rank, item in enumerate(optimized_items[: config.max_items])
        ]

    async def _relevance_first_strategy(
        self, items: List[ContextItem], config: AggregationConfig
    ) -> List[AggregatedItem]:
        """Relevance-first strategy prioritizing query similarity."""
        # Sort purely by relevance score
        sorted_items = sorted(
            items, key=lambda x: x.relevance_score.query_similarity, reverse=True
        )

        return [
            AggregatedItem(
                content=item.content,
                source_info=[f"{item.source_type}:{item.source_id}"],
                relevance_score=item.relevance_score,
                rank=rank + 1,
                is_diverse=False,
                metadata=item.metadata,
            )
            for rank, item in enumerate(sorted_items[: config.max_items])
        ]

    def _get_source_type_weight(self, source_type: SourceType) -> float:
        """Get weight multiplier for different source types."""
        weights = {
            SourceType.GRAPH: 1.2,  # Higher weight for graph sources
            SourceType.VECTOR: 1.0,  # Standard weight for vector sources
            SourceType.DATABASE: 1.1,  # Slightly higher for structured data
            SourceType.HYBRID: 1.0,  # Standard weight for hybrid
        }
        return weights.get(source_type, 1.0)

    async def _calculate_diversity_score(self, items: List[AggregatedItem]) -> float:
        """Calculate overall diversity score for the result set."""
        if len(items) <= 1:
            return 1.0

        total_similarity = 0.0
        comparisons = 0

        for i, item1 in enumerate(items):
            for item2 in items[i + 1 :]:
                # Calculate content similarity if possible
                similarity = 0.0

                # Try to get embeddings from metadata or calculate
                emb1 = self._get_item_embedding(item1)
                emb2 = self._get_item_embedding(item2)

                if emb1 and emb2:
                    similarity = self._calculate_cosine_similarity(emb1, emb2)
                else:
                    # Fallback to simple content overlap
                    content1_words = set(item1.content.lower().split())
                    content2_words = set(item2.content.lower().split())

                    if content1_words and content2_words:
                        overlap = len(content1_words.intersection(content2_words))
                        total_words = len(content1_words.union(content2_words))
                        similarity = overlap / total_words if total_words > 0 else 0.0

                total_similarity += similarity
                comparisons += 1

        # Diversity is inverse of average similarity
        average_similarity = total_similarity / comparisons if comparisons > 0 else 0.0
        diversity_score = 1.0 - average_similarity

        return max(0.0, min(1.0, diversity_score))

    def _get_item_embedding(self, item: AggregatedItem) -> Optional[List[float]]:
        """Extract embedding from aggregated item metadata."""
        return item.metadata.get("embedding")

    async def _update_stats(
        self, result: AggregationResult, strategy: AggregationStrategy
    ) -> None:
        """Update aggregation statistics."""
        self._stats.total_requests += 1

        # Update averages
        if self._stats.total_requests == 1:
            self._stats.average_processing_time_ms = result.aggregation_time_ms
            self._stats.average_items_processed = float(result.total_items_processed)
            self._stats.average_duplicates_removed = float(result.duplicates_removed)
        else:
            # Exponential moving average
            alpha = 0.1
            self._stats.average_processing_time_ms = (
                alpha * result.aggregation_time_ms
                + (1 - alpha) * self._stats.average_processing_time_ms
            )
            self._stats.average_items_processed = (
                alpha * result.total_items_processed
                + (1 - alpha) * self._stats.average_items_processed
            )
            self._stats.average_duplicates_removed = (
                alpha * result.duplicates_removed
                + (1 - alpha) * self._stats.average_duplicates_removed
            )

        # Update strategy usage
        strategy_name = strategy.value
        if strategy_name not in self._stats.strategy_usage_counts:
            self._stats.strategy_usage_counts[strategy_name] = 0
        self._stats.strategy_usage_counts[strategy_name] += 1

        # Update timestamp
        from datetime import datetime, timezone

        self._stats.last_updated = datetime.now(timezone.utc)

    def get_stats(self) -> AggregationStats:
        """Get current aggregation statistics."""
        return self._stats

    async def health_check(self) -> Dict[str, any]:
        """Perform health check of the aggregation system."""
        health_status = {
            "status": "healthy",
            "embedding_model_available": self._embedding_model is not None,
            "embedding_model_name": self.embedding_model_name,
            "sentence_transformers_available": SENTENCE_TRANSFORMERS_AVAILABLE,
            "total_requests_processed": self._stats.total_requests,
            "average_processing_time_ms": self._stats.average_processing_time_ms,
        }

        # Test embedding model if available
        if self._embedding_model:
            try:
                test_embedding = self._embedding_model.encode(
                    ["test"], convert_to_numpy=True
                )
                health_status["embedding_test"] = "passed"
                health_status["embedding_dimension"] = len(test_embedding[0])
            except Exception as e:
                health_status["embedding_test"] = f"failed: {e}"
                health_status["status"] = "degraded"

        return health_status
