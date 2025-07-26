"""
Invisible Learning Middleware for OmniRAG Self-Improvement.

This module provides transparent query learning and adaptation that happens
automatically during normal MCP tool operations. Users never see the learning
system - they just get progressively better results over time.
"""

import asyncio
import logging
import hashlib
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Callable
from functools import wraps
from uuid import UUID

from ..models.session_models import (
    QuerySession,
    QueryInteraction,
    FeedbackType,
    AdaptationStrategy,
)
from ..utils.redis_session import SessionManager
from ..plugins.incremental_learning import LearningOrchestrator


logger = logging.getLogger(__name__)


class InvisibleLearningMiddleware:
    """
    Invisible learning middleware that enhances MCP tools automatically.

    Provides self-healing and self-improving capabilities by:
    - Automatically tracking user interactions
    - Inferring feedback from user behavior patterns
    - Adapting strategies transparently
    - Enhancing tool responses over time

    Users never interact with this directly - it operates behind the scenes.
    """

    def __init__(
        self,
        session_manager: Optional[SessionManager] = None,
        learning_orchestrator: Optional[LearningOrchestrator] = None,
        enable_learning: bool = True,
    ):
        """
        Initialize invisible learning middleware.

        Args:
            session_manager: Redis session manager (auto-created if None)
            learning_orchestrator: ML learning system (auto-created if None)
            enable_learning: Enable/disable learning (for testing)
        """
        self.session_manager = session_manager or SessionManager()
        self.learning_orchestrator = learning_orchestrator or LearningOrchestrator()
        self.enable_learning = enable_learning

        # Behavior tracking for implicit feedback
        self.interaction_history: Dict[str, List[Dict[str, Any]]] = {}
        self.response_times: Dict[str, float] = {}
        self.session_contexts: Dict[str, Dict[str, Any]] = {}

        # Learning thresholds
        self.quick_response_threshold = 2.0  # seconds
        self.engagement_threshold = 30.0  # seconds
        self.reformulation_window = 300.0  # 5 minutes

        logger.info("Invisible learning middleware initialized")

    def _generate_session_id(self, user_context: Dict[str, Any]) -> str:
        """
        Generate consistent session ID from user context.

        Uses user fingerprinting (IP, user agent, etc.) to maintain
        sessions transparently without explicit session management.
        """
        # Create fingerprint from available context
        fingerprint_data = {
            "user_id": user_context.get("user_id"),
            "ip_address": user_context.get("ip_address"),
            "user_agent": user_context.get("user_agent"),
            "client_id": user_context.get("client_id"),
        }

        # Remove None values and create hash
        clean_data = {k: v for k, v in fingerprint_data.items() if v is not None}
        fingerprint_str = str(sorted(clean_data.items()))
        session_hash = hashlib.sha256(fingerprint_str.encode()).hexdigest()[:16]

        return f"auto_{session_hash}"

    def _infer_feedback_from_behavior(
        self,
        interaction_id: str,
        response_time: float,
        follow_up_query: Optional[str] = None,
        engagement_time: Optional[float] = None,
    ) -> tuple[FeedbackType, float]:
        """
        Infer user feedback from behavioral patterns.

        Args:
            interaction_id: Unique interaction identifier
            response_time: Time to generate response
            follow_up_query: Next query from user (if any)
            engagement_time: Time user spent with response

        Returns:
            Tuple of (feedback_type, confidence_score)
        """
        feedback_signals = []

        # Fast response usually indicates good performance
        if response_time < self.quick_response_threshold:
            feedback_signals.append(("positive", 0.3))

        # Long engagement suggests satisfying response
        if engagement_time and engagement_time > self.engagement_threshold:
            feedback_signals.append(("positive", 0.7))

        # Quick follow-up might indicate insufficient response
        if engagement_time and engagement_time < 5.0:
            feedback_signals.append(("negative", 0.4))

        # Query reformulation suggests initial response was inadequate
        if follow_up_query:
            # Simple heuristic: if follow-up is very similar, might be reformulation
            if len(follow_up_query) > 10:  # Not just "thanks" or "ok"
                feedback_signals.append(("negative", 0.5))

        # No follow-up within reasonable time = positive
        if not follow_up_query and engagement_time is None:
            # Default assumption: no news is good news
            feedback_signals.append(("positive", 0.2))

        # Aggregate signals
        if not feedback_signals:
            return FeedbackType.NEUTRAL, 0.5

        positive_weight = sum(
            score for signal, score in feedback_signals if signal == "positive"
        )
        negative_weight = sum(
            score for signal, score in feedback_signals if signal == "negative"
        )

        if positive_weight > negative_weight:
            return FeedbackType.POSITIVE, min(positive_weight, 1.0)
        elif negative_weight > positive_weight:
            return FeedbackType.NEGATIVE, min(negative_weight, 1.0)
        else:
            return FeedbackType.NEUTRAL, 0.5

    async def _get_or_create_session(
        self, user_context: Dict[str, Any]
    ) -> QuerySession:
        """Get or create session transparently."""
        session_id = self._generate_session_id(user_context)

        # Try to get existing session
        session = await self.session_manager.get_session(session_id)

        if not session:
            # Create new session transparently
            session = await self.session_manager.create_session(
                user_id=user_context.get("user_id"),
                session_id=UUID(int=int(session_id.replace("auto_", ""), 16)),
            )
            logger.debug("Created invisible session for user context")

        return session

    async def _track_interaction_invisibly(
        self,
        query: str,
        response: Any,
        user_context: Dict[str, Any],
        tool_name: str,
        response_time: float,
    ) -> None:
        """Track interaction without user awareness."""
        if not self.enable_learning:
            return

        try:
            # Get session transparently
            session = await self._get_or_create_session(user_context)

            # Create interaction record
            interaction = QueryInteraction(
                query=query,
                context={
                    "tool_name": tool_name,
                    "user_context": user_context,
                    "response_preview": str(response)[:100] if response else None,
                },
                response=str(response) if response else None,
                response_time_ms=response_time * 1000,
            )

            # Store for behavior analysis
            interaction_key = str(interaction.interaction_id)
            self.interaction_history[interaction_key] = {
                "timestamp": datetime.now(timezone.utc),
                "session_id": str(session.session_id),
                "tool_name": tool_name,
                "query": query,
            }
            self.response_times[interaction_key] = response_time

            # Add to session
            session.add_interaction(interaction)

            # Save session
            await self.session_manager.save_session(session)

            # Trigger background learning
            asyncio.create_task(self._background_learning(session, interaction))

        except Exception as e:
            logger.warning(f"Error in invisible interaction tracking: {e}")

    async def _background_learning(
        self, session: QuerySession, interaction: QueryInteraction
    ) -> None:
        """Perform learning in background without blocking user experience."""
        try:
            # Infer feedback from accumulated behavior
            feedback_type, confidence = self._infer_feedback_from_behavior(
                str(interaction.interaction_id),
                (
                    interaction.response_time_ms / 1000
                    if interaction.response_time_ms
                    else 0
                ),
            )

            # Update interaction with inferred feedback
            interaction.feedback_type = feedback_type
            interaction.feedback_value = confidence

            # Learn from interaction
            await self.learning_orchestrator.learn_from_interaction(interaction)

            # Check if adaptation is needed and do it transparently
            if session.should_adapt():
                await self._adapt_transparently(session)

            # Save updated models
            model_states = self.learning_orchestrator.get_model_states()
            for model_name, model_state in model_states.items():
                await self.session_manager.save_model_state(
                    session.session_id, model_name, model_state
                )

        except Exception as e:
            logger.warning(f"Error in background learning: {e}")

    async def _adapt_transparently(self, session: QuerySession) -> None:
        """Adapt strategies transparently without user awareness."""
        try:
            feedback_ratio = session.get_feedback_ratio()

            # Select adaptation strategy based on patterns
            if feedback_ratio < 0.2:
                new_strategy = AdaptationStrategy.PATTERN_BASED
            elif feedback_ratio < 0.4:
                new_strategy = AdaptationStrategy.CONTEXT_AWARE
            else:
                new_strategy = AdaptationStrategy.HYBRID

            # Apply adaptation
            old_strategy = session.current_strategy
            session.current_strategy = new_strategy
            session.metrics.adaptation_frequency += 1
            session.metrics.last_adaptation = datetime.now(timezone.utc)

            # Log adaptation (for debugging only)
            logger.info(
                f"Transparent adaptation: {old_strategy.value} -> {new_strategy.value}"
            )

            # Save adapted session
            await self.session_manager.save_session(session)

        except Exception as e:
            logger.warning(f"Error in transparent adaptation: {e}")

    async def enhance_tool_response(
        self, response: Any, user_context: Dict[str, Any], tool_name: str
    ) -> Any:
        """
        Enhance tool response based on learned preferences.

        This is where the learning system improves user experience
        transparently by adapting responses to learned preferences.
        """
        if not self.enable_learning:
            return response

        try:
            # Get session for learned preferences
            session = await self._get_or_create_session(user_context)

            # Load session-specific models
            model_states = {}
            for model_name in ["query_classifier", "preference_regressor"]:
                state = await self.session_manager.get_model_state(
                    session.session_id, model_name
                )
                if state:
                    model_states[model_name] = state

            if model_states:
                self.learning_orchestrator.load_model_states(model_states)

            # Apply learned enhancements (this is where the magic happens)
            enhanced_response = await self._apply_learned_enhancements(
                response, session, tool_name
            )

            return enhanced_response

        except Exception as e:
            logger.warning(f"Error enhancing tool response: {e}")
            return response  # Fallback to original response

    async def _apply_learned_enhancements(
        self, response: Any, session: QuerySession, tool_name: str
    ) -> Any:
        """Apply learned enhancements to tool response."""
        # This is where specific tool enhancements happen based on learning

        # Example enhancements based on tool type:
        if tool_name == "context_aggregation":
            # Enhance context aggregation based on learned preferences
            return await self._enhance_context_aggregation(response, session)

        elif tool_name == "query_routing":
            # Enhance routing decisions based on success patterns
            return await self._enhance_query_routing(response, session)

        # Default: return response as-is
        return response

    async def _enhance_context_aggregation(
        self, response: Any, session: QuerySession
    ) -> Any:
        """Enhance context aggregation based on learned user preferences."""
        # Apply user preference weights to context aggregation
        preferences = session.user_preferences

        if hasattr(response, "contexts") and preferences.context_preferences:
            # Weight contexts based on learned preferences
            for context in response.contexts:
                context_type = context.get("type", "default")
                preference_weight = preferences.context_preferences.get(
                    context_type, 1.0
                )
                if hasattr(context, "weight"):
                    context.weight *= preference_weight

        return response

    async def _enhance_query_routing(self, response: Any, session: QuerySession) -> Any:
        """Enhance query routing based on learned success patterns."""
        # Adjust routing confidence based on historical success
        strategy_history = session.strategy_history

        if strategy_history and hasattr(response, "confidence"):
            # Boost confidence for strategies that have worked well
            current_strategy = session.current_strategy
            strategy_success = self._calculate_strategy_success(
                strategy_history, current_strategy
            )
            response.confidence *= strategy_success

        return response

    def _calculate_strategy_success(
        self, history: List[Dict], strategy: AdaptationStrategy
    ) -> float:
        """Calculate success rate for a given strategy."""
        strategy_uses = [h for h in history if h.get("new_strategy") == strategy.value]
        if not strategy_uses:
            return 1.0  # No data, assume neutral

        # Simple success heuristic based on feedback ratio improvement
        success_count = sum(
            1 for use in strategy_uses if use.get("feedback_ratio", 0.5) > 0.5
        )
        return min(1.5, max(0.5, success_count / len(strategy_uses)))


def invisible_learning_decorator(middleware: InvisibleLearningMiddleware):
    """
    Decorator to add invisible learning to MCP tools.

    This decorator wraps existing MCP tools to add automatic learning
    without changing their interface or user experience.
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = datetime.now(timezone.utc)

            # Extract query and user context from arguments
            query = kwargs.get("query", str(args[0]) if args else "unknown")
            user_context = kwargs.get("user_context", {})
            tool_name = func.__name__

            try:
                # Execute original tool
                response = await func(*args, **kwargs)

                # Calculate response time
                response_time = (
                    datetime.now(timezone.utc) - start_time
                ).total_seconds()

                # Enhance response with learned improvements
                enhanced_response = await middleware.enhance_tool_response(
                    response, user_context, tool_name
                )

                # Track interaction invisibly (fire and forget)
                asyncio.create_task(
                    middleware._track_interaction_invisibly(
                        query, enhanced_response, user_context, tool_name, response_time
                    )
                )

                return enhanced_response

            except Exception as e:
                # Track errors for learning too
                response_time = (
                    datetime.now(timezone.utc) - start_time
                ).total_seconds()
                asyncio.create_task(
                    middleware._track_interaction_invisibly(
                        query, f"Error: {e}", user_context, tool_name, response_time
                    )
                )
                raise

        return wrapper

    return decorator


# Global middleware instance (singleton pattern)
_global_middleware: Optional[InvisibleLearningMiddleware] = None


def get_learning_middleware() -> InvisibleLearningMiddleware:
    """Get or create global learning middleware instance."""
    global _global_middleware
    if _global_middleware is None:
        _global_middleware = InvisibleLearningMiddleware()
    return _global_middleware


def enable_invisible_learning():
    """
    Enable invisible learning for all MCP tools.

    This should be called once during system initialization to enable
    automatic learning and improvement across all tools.
    """
    middleware = get_learning_middleware()
    logger.info("Invisible learning enabled - OmniRAG will improve automatically")
    return middleware


def disable_invisible_learning():
    """Disable invisible learning (for testing or maintenance)."""
    global _global_middleware
    if _global_middleware:
        _global_middleware.enable_learning = False
    logger.info("Invisible learning disabled")


async def get_learning_diagnostics() -> Dict[str, Any]:
    """
    Get learning system diagnostics (admin tool only).

    This is the ONLY exposed function for system administrators
    to monitor the invisible learning system.
    """
    middleware = get_learning_middleware()

    try:
        # Get basic stats
        stats = {
            "learning_enabled": middleware.enable_learning,
            "tracked_interactions": len(middleware.interaction_history),
            "active_sessions": len(middleware.session_contexts),
            "average_response_time": sum(middleware.response_times.values())
            / max(1, len(middleware.response_times)),
            "system_health": "healthy",
        }

        # Get session cleanup stats
        cleanup_stats = await middleware.session_manager.cleanup_expired_sessions()
        stats["cleanup_count"] = cleanup_stats

        return stats

    except Exception as e:
        return {"learning_enabled": False, "error": str(e), "system_health": "degraded"}
