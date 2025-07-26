"""
Session-Aware Query Learning and Adaptation System (OMR-P3-004).

This module implements the core query learning system that tracks user interactions,
adapts query strategies based on feedback, and maintains session-specific learning state.
Integrates with Redis for session persistence and scikit-learn for incremental learning.
"""

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union

from mcp.server import Server
from mcp.types import Tool, TextContent
from pydantic import BaseModel, Field

from ..models.session_models import (
    QuerySession,
    QueryInteraction,
    FeedbackType,
    AdaptationStrategy,
)
from ..utils.redis_session import SessionManager
from ..plugins.incremental_learning import LearningOrchestrator
from ..plugins.context_aggregator import ContextAggregator
from ..config.settings import get_settings


logger = logging.getLogger(__name__)


class QueryLearningError(Exception):
    """Query learning system errors."""

    pass


class QueryLearningRequest(BaseModel):
    """Request model for query learning operations."""

    session_id: Optional[str] = Field(None, description="Session identifier")
    user_id: Optional[str] = Field(None, description="User identifier")
    query: str = Field(..., description="User query text")
    context: Dict[str, Any] = Field(default_factory=dict, description="Query context")
    feedback_type: Optional[FeedbackType] = Field(
        None, description="User feedback type"
    )
    feedback_value: Optional[Union[int, float, str]] = Field(
        None, description="Feedback value"
    )
    response: Optional[str] = Field(None, description="System response")


class AdaptationRequest(BaseModel):
    """Request model for strategy adaptation."""

    session_id: str = Field(..., description="Session identifier")
    force_adaptation: bool = Field(
        False, description="Force adaptation regardless of thresholds"
    )
    strategy: Optional[AdaptationStrategy] = Field(
        None, description="Specific strategy to apply"
    )


class SessionInsightsRequest(BaseModel):
    """Request model for session insights."""

    session_id: str = Field(..., description="Session identifier")
    include_interactions: bool = Field(False, description="Include interaction history")
    include_model_states: bool = Field(
        False, description="Include model state information"
    )


class QueryLearningSystem:
    """
    Main query learning and adaptation system.

    Provides session-aware learning capabilities with Redis persistence,
    incremental ML models, and integration with context aggregation.
    """

    def __init__(
        self,
        session_manager: Optional[SessionManager] = None,
        context_aggregator: Optional[ContextAggregator] = None,
        adaptation_threshold: float = 0.3,
        min_interactions_for_adaptation: int = 5,
    ):
        """
        Initialize query learning system.

        Args:
            session_manager: Redis session manager instance
            context_aggregator: Context aggregation system
            adaptation_threshold: Minimum feedback ratio to trigger adaptation
            min_interactions_for_adaptation: Minimum interactions before adaptation
        """
        self.settings = get_settings()
        self.session_manager = session_manager or SessionManager()
        self.context_aggregator = context_aggregator
        self.learning_orchestrator = LearningOrchestrator()

        # Adaptation parameters
        self.adaptation_threshold = adaptation_threshold
        self.min_interactions = min_interactions_for_adaptation

        # Performance tracking
        self.total_interactions = 0
        self.total_adaptations = 0
        self.last_cleanup = datetime.now(timezone.utc)

        logger.info("QueryLearningSystem initialized")

    async def initialize(self) -> None:
        """Initialize system components."""
        logger.info("Initializing QueryLearningSystem components")

        # Cleanup expired sessions on startup
        await self.session_manager.cleanup_expired_sessions()

        # Initialize context aggregator if not provided
        if self.context_aggregator is None:
            self.context_aggregator = ContextAggregator()
            await self.context_aggregator.initialize()

    async def track_interaction(
        self,
        query: str,
        context: Dict[str, Any],
        response: Optional[str] = None,
        feedback_type: Optional[FeedbackType] = None,
        feedback_value: Optional[Union[int, float, str]] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Track a user query interaction and update learning models.

        Args:
            query: User query text
            context: Query context information
            response: System response (if available)
            feedback_type: Type of user feedback
            feedback_value: Feedback value/rating
            session_id: Session identifier (creates new if None)
            user_id: User identifier

        Returns:
            Interaction tracking results
        """
        try:
            start_time = datetime.now(timezone.utc)

            # Get or create session
            session = await self._get_or_create_session(session_id, user_id)

            # Predict intent using context aggregator
            query_intent = None
            if self.context_aggregator:
                intent_result = await self.context_aggregator.classify_intent(
                    query, context
                )
                query_intent = intent_result.get("predicted_intent")

            # Create interaction record
            interaction = QueryInteraction(
                query=query,
                query_intent=query_intent,
                context=context,
                response=response,
                feedback_type=feedback_type,
                feedback_value=feedback_value,
                response_time_ms=(
                    datetime.now(timezone.utc) - start_time
                ).total_seconds()
                * 1000,
            )

            # Add to session
            session.add_interaction(interaction)

            # Learn from interaction
            learning_results = await self.learning_orchestrator.learn_from_interaction(
                interaction
            )

            # Save updated session
            await self.session_manager.save_session(session)

            # Save model states
            model_states = self.learning_orchestrator.get_model_states()
            for model_name, model_state in model_states.items():
                await self.session_manager.save_model_state(
                    session.session_id, model_name, model_state
                )

            # Check for adaptation need
            adaptation_needed = session.should_adapt(self.adaptation_threshold)

            self.total_interactions += 1

            result = {
                "session_id": str(session.session_id),
                "interaction_id": str(interaction.interaction_id),
                "query_intent": query_intent,
                "learning_results": learning_results,
                "adaptation_needed": adaptation_needed,
                "session_metrics": {
                    "total_interactions": session.metrics.total_interactions,
                    "feedback_ratio": session.get_feedback_ratio(),
                    "current_strategy": session.current_strategy.value,
                },
            }

            logger.debug(f"Tracked interaction for session {session.session_id}")
            return result

        except Exception as e:
            logger.error(f"Error tracking interaction: {e}")
            raise QueryLearningError(f"Failed to track interaction: {e}")

    async def adapt_strategies(
        self,
        session_id: str,
        force_adaptation: bool = False,
        target_strategy: Optional[AdaptationStrategy] = None,
    ) -> Dict[str, Any]:
        """
        Adapt query strategies based on session learning.

        Args:
            session_id: Session identifier
            force_adaptation: Force adaptation regardless of thresholds
            target_strategy: Specific strategy to apply

        Returns:
            Adaptation results and new strategy
        """
        try:
            # Get session
            session = await self.session_manager.get_session(session_id)
            if not session:
                raise QueryLearningError(f"Session {session_id} not found")

            # Check adaptation criteria
            if not force_adaptation and not session.should_adapt(
                self.adaptation_threshold
            ):
                return {
                    "adapted": False,
                    "reason": "Adaptation criteria not met",
                    "current_strategy": session.current_strategy.value,
                    "feedback_ratio": session.get_feedback_ratio(),
                }

            # Determine new strategy
            if target_strategy:
                new_strategy = target_strategy
            else:
                new_strategy = await self._select_adaptation_strategy(session)

            # Apply adaptation
            old_strategy = session.current_strategy
            session.current_strategy = new_strategy
            session.metrics.adaptation_frequency += 1
            session.metrics.last_adaptation = datetime.now(timezone.utc)

            # Record strategy change
            strategy_change = {
                "timestamp": datetime.now(timezone.utc),
                "old_strategy": old_strategy.value,
                "new_strategy": new_strategy.value,
                "feedback_ratio": session.get_feedback_ratio(),
                "interaction_count": session.metrics.total_interactions,
                "forced": force_adaptation,
            }
            session.strategy_history.append(strategy_change)

            # Update user preferences based on adaptation
            await self._update_user_preferences(session, new_strategy)

            # Save updated session
            await self.session_manager.save_session(session)

            self.total_adaptations += 1

            result = {
                "adapted": True,
                "old_strategy": old_strategy.value,
                "new_strategy": new_strategy.value,
                "strategy_change": strategy_change,
                "session_metrics": {
                    "total_interactions": session.metrics.total_interactions,
                    "feedback_ratio": session.get_feedback_ratio(),
                    "adaptation_frequency": session.metrics.adaptation_frequency,
                },
            }

            logger.info(
                f"Adapted strategy for session {session_id}: {old_strategy.value} -> {new_strategy.value}"
            )
            return result

        except Exception as e:
            logger.error(f"Error adapting strategies for session {session_id}: {e}")
            raise QueryLearningError(f"Failed to adapt strategies: {e}")

    async def get_session_insights(
        self,
        session_id: str,
        include_interactions: bool = False,
        include_model_states: bool = False,
    ) -> Dict[str, Any]:
        """
        Get learning insights for a session.

        Args:
            session_id: Session identifier
            include_interactions: Include interaction history
            include_model_states: Include model state information

        Returns:
            Session insights and analytics
        """
        try:
            # Get session
            session = await self.session_manager.get_session(session_id)
            if not session:
                raise QueryLearningError(f"Session {session_id} not found")

            # Basic session info
            insights = {
                "session_id": str(session.session_id),
                "user_id": session.user_id,
                "status": session.status.value,
                "created_at": session.created_at.isoformat(),
                "last_activity": session.last_activity.isoformat(),
                "current_strategy": session.current_strategy.value,
                # Metrics
                "metrics": {
                    "total_interactions": session.metrics.total_interactions,
                    "positive_feedback_count": session.metrics.positive_feedback_count,
                    "negative_feedback_count": session.metrics.negative_feedback_count,
                    "feedback_ratio": session.get_feedback_ratio(),
                    "average_response_time_ms": session.metrics.average_response_time_ms,
                    "adaptation_frequency": session.metrics.adaptation_frequency,
                    "last_adaptation": (
                        session.metrics.last_adaptation.isoformat()
                        if session.metrics.last_adaptation
                        else None
                    ),
                },
                # User preferences
                "user_preferences": session.user_preferences.model_dump(),
                # Strategy history
                "strategy_history": session.strategy_history,
            }

            # Add interactions if requested
            if include_interactions:
                insights["interactions"] = [
                    interaction.model_dump(mode="json")
                    for interaction in session.interactions[
                        -50:
                    ]  # Last 50 interactions
                ]

            # Add model states if requested
            if include_model_states:
                model_states = {}
                for model_name in ["query_classifier", "preference_regressor"]:
                    state = await self.session_manager.get_model_state(
                        session_id, model_name
                    )
                    if state:
                        model_states[model_name] = {
                            "model_type": state.model_type,
                            "model_version": state.model_version,
                            "last_updated": state.last_updated.isoformat(),
                            "update_count": state.update_count,
                            "feature_count": len(state.feature_names),
                        }
                insights["model_states"] = model_states

            return insights

        except Exception as e:
            logger.error(f"Error getting session insights for {session_id}: {e}")
            raise QueryLearningError(f"Failed to get session insights: {e}")

    async def predict_preferences(
        self, queries: List[str], session_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Predict user preferences for queries.

        Args:
            queries: List of query texts
            session_id: Optional session identifier for context

        Returns:
            List of preference predictions
        """
        try:
            # Load session-specific models if available
            if session_id:
                await self._load_session_models(session_id)

            # Get predictions from learning orchestrator
            intent_predictions = await self.learning_orchestrator.predict_intent(
                queries
            )
            preference_predictions = (
                await self.learning_orchestrator.predict_preferences(queries)
            )

            # Combine results
            results = []
            for i, query in enumerate(queries):
                result = {
                    "query": query,
                    "intent_prediction": (
                        intent_predictions[i] if i < len(intent_predictions) else {}
                    ),
                    "preference_prediction": (
                        preference_predictions[i]
                        if i < len(preference_predictions)
                        else {}
                    ),
                }
                results.append(result)

            return results

        except Exception as e:
            logger.error(f"Error predicting preferences: {e}")
            return []

    async def _get_or_create_session(
        self, session_id: Optional[str], user_id: Optional[str]
    ) -> QuerySession:
        """Get existing session or create new one."""
        if session_id:
            session = await self.session_manager.get_session(session_id)
            if session:
                return session

        # Create new session
        return await self.session_manager.create_session(user_id=user_id)

    async def _select_adaptation_strategy(
        self, session: QuerySession
    ) -> AdaptationStrategy:
        """Select appropriate adaptation strategy based on session patterns."""
        feedback_ratio = session.get_feedback_ratio()
        interaction_count = session.metrics.total_interactions

        # Strategy selection logic
        if feedback_ratio < 0.2:  # Very poor feedback
            # Try pattern-based learning
            return AdaptationStrategy.PATTERN_BASED
        elif feedback_ratio < 0.4:  # Poor feedback
            # Try context-aware adaptation
            return AdaptationStrategy.CONTEXT_AWARE
        elif interaction_count > 20:  # Enough data for hybrid approach
            return AdaptationStrategy.HYBRID
        else:
            # Default to feature-based for smaller datasets
            return AdaptationStrategy.FEATURE_BASED

    async def _update_user_preferences(
        self, session: QuerySession, new_strategy: AdaptationStrategy
    ) -> None:
        """Update user preferences based on adaptation."""
        preferences = session.user_preferences

        # Add strategy to preferred strategies
        if new_strategy not in preferences.preferred_strategies:
            preferences.preferred_strategies.append(new_strategy)

        # Analyze recent interactions for pattern updates
        recent_interactions = session.interactions[-10:]  # Last 10 interactions

        if recent_interactions:
            # Update query patterns
            query_patterns = [interaction.query for interaction in recent_interactions]
            preferences.query_patterns.extend(
                query_patterns[-5:]
            )  # Keep recent patterns

            # Update context preferences
            for interaction in recent_interactions:
                if interaction.feedback_type == FeedbackType.POSITIVE:
                    for context_key, context_value in interaction.context.items():
                        if isinstance(context_value, str):
                            current_weight = preferences.context_preferences.get(
                                context_key, 0.0
                            )
                            preferences.context_preferences[context_key] = min(
                                1.0, current_weight + 0.1
                            )

    async def _load_session_models(self, session_id: str) -> None:
        """Load session-specific model states."""
        try:
            model_states = {}
            for model_name in ["query_classifier", "preference_regressor"]:
                state = await self.session_manager.get_model_state(
                    session_id, model_name
                )
                if state:
                    model_states[model_name] = state

            if model_states:
                self.learning_orchestrator.load_model_states(model_states)
                logger.debug(
                    f"Loaded {len(model_states)} model states for session {session_id}"
                )

        except Exception as e:
            logger.warning(f"Failed to load session models for {session_id}: {e}")

    async def cleanup_expired_sessions(self) -> Dict[str, Any]:
        """Clean up expired sessions and return cleanup statistics."""
        try:
            cleanup_count = await self.session_manager.cleanup_expired_sessions()
            self.last_cleanup = datetime.now(timezone.utc)

            return {
                "cleanup_count": cleanup_count,
                "last_cleanup": self.last_cleanup.isoformat(),
                "total_interactions": self.total_interactions,
                "total_adaptations": self.total_adaptations,
            }

        except Exception as e:
            logger.error(f"Error during session cleanup: {e}")
            return {"error": str(e)}

    async def get_system_stats(self) -> Dict[str, Any]:
        """Get system-wide statistics and health metrics."""
        return {
            "total_interactions": self.total_interactions,
            "total_adaptations": self.total_adaptations,
            "adaptation_rate": self.total_adaptations / max(1, self.total_interactions),
            "last_cleanup": self.last_cleanup.isoformat(),
            "settings": {
                "adaptation_threshold": self.adaptation_threshold,
                "min_interactions": self.min_interactions,
            },
        }


def register_query_learning_tools(
    server: Server, query_learning_system: QueryLearningSystem
) -> None:
    """Register MCP tools for query learning system."""

    @server.list_tools()
    async def list_query_learning_tools() -> List[Tool]:
        return [
            Tool(
                name="track_query_interaction",
                description="Track a user query interaction for learning and adaptation",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "User query text"},
                        "context": {"type": "object", "description": "Query context"},
                        "response": {
                            "type": "string",
                            "description": "System response",
                        },
                        "feedback_type": {
                            "type": "string",
                            "enum": [
                                "positive",
                                "negative",
                                "neutral",
                                "explicit_rating",
                            ],
                        },
                        "feedback_value": {
                            "type": ["number", "string"],
                            "description": "Feedback value or rating",
                        },
                        "session_id": {
                            "type": "string",
                            "description": "Session identifier",
                        },
                        "user_id": {"type": "string", "description": "User identifier"},
                    },
                    "required": ["query", "context"],
                },
            ),
            Tool(
                name="adapt_query_strategies",
                description="Adapt query strategies based on session learning",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "session_id": {
                            "type": "string",
                            "description": "Session identifier",
                        },
                        "force_adaptation": {
                            "type": "boolean",
                            "description": "Force adaptation",
                        },
                        "target_strategy": {
                            "type": "string",
                            "enum": [
                                "feature_based",
                                "pattern_based",
                                "context_aware",
                                "hybrid",
                            ],
                        },
                    },
                    "required": ["session_id"],
                },
            ),
            Tool(
                name="get_session_insights",
                description="Get learning insights and analytics for a session",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "session_id": {
                            "type": "string",
                            "description": "Session identifier",
                        },
                        "include_interactions": {
                            "type": "boolean",
                            "description": "Include interaction history",
                        },
                        "include_model_states": {
                            "type": "boolean",
                            "description": "Include model state information",
                        },
                    },
                    "required": ["session_id"],
                },
            ),
            Tool(
                name="predict_query_preferences",
                description="Predict user preferences for given queries",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "queries": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of queries",
                        },
                        "session_id": {
                            "type": "string",
                            "description": "Session identifier for context",
                        },
                    },
                    "required": ["queries"],
                },
            ),
            Tool(
                name="get_query_learning_stats",
                description="Get system-wide query learning statistics",
                inputSchema={"type": "object", "properties": {}, "required": []},
            ),
        ]

    @server.call_tool()
    async def call_query_learning_tool(name: str, arguments: dict) -> List[TextContent]:
        try:
            if name == "track_query_interaction":
                result = await query_learning_system.track_interaction(
                    query=arguments["query"],
                    context=arguments.get("context", {}),
                    response=arguments.get("response"),
                    feedback_type=(
                        FeedbackType(arguments["feedback_type"])
                        if arguments.get("feedback_type")
                        else None
                    ),
                    feedback_value=arguments.get("feedback_value"),
                    session_id=arguments.get("session_id"),
                    user_id=arguments.get("user_id"),
                )

            elif name == "adapt_query_strategies":
                result = await query_learning_system.adapt_strategies(
                    session_id=arguments["session_id"],
                    force_adaptation=arguments.get("force_adaptation", False),
                    target_strategy=(
                        AdaptationStrategy(arguments["target_strategy"])
                        if arguments.get("target_strategy")
                        else None
                    ),
                )

            elif name == "get_session_insights":
                result = await query_learning_system.get_session_insights(
                    session_id=arguments["session_id"],
                    include_interactions=arguments.get("include_interactions", False),
                    include_model_states=arguments.get("include_model_states", False),
                )

            elif name == "predict_query_preferences":
                result = await query_learning_system.predict_preferences(
                    queries=arguments["queries"], session_id=arguments.get("session_id")
                )

            elif name == "get_query_learning_stats":
                result = await query_learning_system.get_system_stats()

            else:
                return [TextContent(type="text", text=f"Unknown tool: {name}")]

            return [TextContent(type="text", text=str(result))]

        except Exception as e:
            logger.error(f"Error in {name}: {e}")
            return [TextContent(type="text", text=f"Error: {e}")]
