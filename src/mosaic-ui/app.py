#!/usr/bin/env python3
"""
Mosaic Knowledge Graph UI - Streamlit Application
Interactive graph visualization and AI-powered chat interface using real Cosmos DB data and Semantic Kernel
"""

import streamlit as st
import streamlit_agraph as agraph
from streamlit_agraph import agraph, Node, Edge, Config
import asyncio
import logging
import os
import sys
from typing import Dict, List, Any
from pathlib import Path
import pandas as pd
import plotly.express as px
import uuid

# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    # Load .env from project root (two levels up from src/mosaic-ui/app.py)
    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✅ Loaded environment variables from {env_path}")
    else:
        print(f"⚠️ No .env file found at {env_path}")
except ImportError:
    print("⚠️ python-dotenv not available, relying on system environment variables")

# Configure comprehensive logging for both Streamlit and Semantic Kernel
logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s - %(name)s:%(lineno)d - %(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Set up specific loggers
logger = logging.getLogger(__name__)
sk_logger = logging.getLogger("semantic_kernel")
cosmos_logger = logging.getLogger("azure.cosmos")

# Enable debug logging for key components
sk_logger.setLevel(logging.DEBUG)
cosmos_logger.setLevel(logging.DEBUG)

# Enable Semantic Kernel telemetry (both regular and sensitive data)
os.environ["SEMANTICKERNEL_EXPERIMENTAL_GENAI_ENABLE_OTEL_DIAGNOSTICS"] = "true"
os.environ["SEMANTICKERNEL_EXPERIMENTAL_GENAI_ENABLE_OTEL_DIAGNOSTICS_SENSITIVE"] = (
    "true"
)

logger.info("🎯 Starting Mosaic Knowledge Graph UI with comprehensive logging enabled")

# Add the parent directories to the path to import mosaic modules
sys.path.append(str(Path(__file__).parent.parent / "mosaic-ingestion"))
sys.path.append(str(Path(__file__).parent.parent / "mosaic-mcp"))

# Import Mosaic components for real database integration
try:
    from cosmos_mode_manager import CosmosModeManager
    from azure.cosmos import CosmosClient
    from azure.cosmos.exceptions import (
        CosmosHttpResponseError,
        CosmosResourceNotFoundError,
    )

    logger.info("✅ Cosmos DB components successfully imported")
    COSMOS_AVAILABLE = True
except ImportError as e:
    error_msg = f"Cosmos components not available: {e}"
    logger.error(error_msg)
    st.error(f"❌ Database Error: {error_msg}")
    st.exception(e)
    COSMOS_AVAILABLE = False

try:
    import semantic_kernel as sk
    from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
    from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
    from semantic_kernel.prompt_template import PromptTemplateConfig
    from semantic_kernel.core_plugins.text_memory_plugin import TextMemoryPlugin
    from semantic_kernel.memory.null_memory import NullMemory

    logger.info("✅ Semantic Kernel components successfully imported")
    SEMANTIC_KERNEL_AVAILABLE = True
except ImportError as e:
    error_msg = f"Semantic Kernel not available: {e}"
    logger.error(error_msg)
    st.warning(f"⚠️ AI Chat Unavailable: {error_msg}")
    SEMANTIC_KERNEL_AVAILABLE = False

# Configure logging
logger.info("🚀 All imports processed, configuring application...")

# Set Streamlit page config
st.set_page_config(
    page_title="Mosaic Knowledge Graph",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Award-winning Dark Mode CSS - Material Design 3 + GitHub-inspired
st.markdown(
    """
<style>
    /* === DARK MODE FOUNDATION === */
    .stApp {
        background-color: #0D1117;
        color: #F0F6FC;
    }
    
    /* === HEADER STYLING === */
    .main-header {
        background: linear-gradient(135deg, #238636 0%, #2EA043 100%);
        padding: 2rem;
        border-radius: 12px;
        color: #F0F6FC;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(35,134,54,0.2);
        border: 1px solid #2EA043;
    }
    
    /* === CARD COMPONENTS === */
    .metric-card {
        background: #161B22;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #58A6FF;
        margin: 0.5rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        border: 1px solid #30363D;
        color: #F0F6FC;
    }
    
    .data-status {
        background: #161B22;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #3FB950;
        border: 1px solid #2EA043;
        color: #F0F6FC;
    }
    
    .error-status {
        background: #161B22;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #F85149;
        border: 1px solid #DA3633;
        color: #F0F6FC;
    }
    
    .chat-message {
        background: #21262D;
        padding: 1rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        border: 1px solid #30363D;
        color: #F0F6FC;
    }
    
    .user-message {
        background: #161B22;
        margin-left: 2rem;
        border-left: 4px solid #3FB950;
        border: 1px solid #2EA043;
        color: #F0F6FC;
    }
    
    .assistant-message {
        background: #161B22;
        margin-right: 2rem;
        border-left: 4px solid #D29922;
        border: 1px solid #BB8009;
        color: #F0F6FC;
    }
    
    .graph-controls {
        background: #21262D;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid #30363D;
        color: #F0F6FC;
    }
    
    /* === STREAMLIT COMPONENT OVERRIDES === */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: #161B22;
        border-radius: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #21262D;
        border-radius: 8px 8px 0 0;
        padding: 10px 24px;
        color: #8B949E;
        border: 1px solid #30363D;
        transition: all 0.2s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #30363D;
        color: #F0F6FC;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #238636;
        color: #F0F6FC;
        border-color: #2EA043;
        box-shadow: 0 4px 16px rgba(35,134,54,0.2);
    }
    
    /* === INTERACTIVE ELEMENTS === */
    .stButton > button {
        background: linear-gradient(135deg, #238636 0%, #2EA043 100%);
        color: #F0F6FC;
        border: 1px solid #2EA043;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.2s ease;
        box-shadow: 0 4px 16px rgba(35,134,54,0.2);
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #2EA043 0%, #3FB950 100%);
        box-shadow: 0 6px 24px rgba(35,134,54,0.3);
        transform: translateY(-1px);
    }
    
    /* === INPUT FIELDS === */
    .stTextInput > div > div > input {
        background-color: #21262D;
        color: #F0F6FC;
        border: 1px solid #30363D;
        border-radius: 8px;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #58A6FF;
        box-shadow: 0 0 0 3px rgba(88,166,255,0.1);
    }
    
    .stSelectbox > div > div > select {
        background-color: #21262D;
        color: #F0F6FC;
        border: 1px solid #30363D;
        border-radius: 8px;
    }
    
    /* === TEXT STYLING === */
    .stMarkdown {
        color: #F0F6FC;
    }
    
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, 
    .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
        color: #F0F6FC;
    }
    
    .stMarkdown a {
        color: #58A6FF;
    }
    
    .stMarkdown a:hover {
        color: #79C0FF;
    }
    
    /* === SIDEBAR === */
    .css-1d391kg {
        background-color: #161B22;
        border-right: 1px solid #30363D;
    }
    
    .css-1d391kg .stMarkdown {
        color: #F0F6FC;
    }
    
    /* === METRICS === */
    .stMetric {
        background-color: #21262D;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #30363D;
    }
    
    .stMetric > div {
        color: #F0F6FC;
    }
    
    /* === EXPANDERS === */
    .streamlit-expanderHeader {
        background-color: #21262D;
        color: #F0F6FC;
        border: 1px solid #30363D;
        border-radius: 8px;
    }
    
    .streamlit-expanderContent {
        background-color: #161B22;
        border: 1px solid #30363D;
        border-radius: 0 0 8px 8px;
    }
    
    /* === DATAFRAMES === */
    .stDataFrame {
        background-color: #21262D;
        border: 1px solid #30363D;
        border-radius: 8px;
    }
    
    /* === PROGRESS BARS === */
    .stProgress .st-bo {
        background-color: #21262D;
    }
    
    .stProgress .st-bp {
        background-color: #3FB950;
    }
    
    /* === ALERTS === */
    .stAlert {
        background-color: #161B22;
        border: 1px solid #30363D;
        color: #F0F6FC;
    }
    
    .stSuccess {
        border-left: 4px solid #3FB950;
    }
    
    .stWarning {
        border-left: 4px solid #D29922;
    }
    
    .stError {
        border-left: 4px solid #F85149;
    }
    
    .stInfo {
        border-left: 4px solid #58A6FF;
    }
</style>
""",
    unsafe_allow_html=True,
)


class MosaicDataService:
    """Service for loading and managing Mosaic knowledge graph data."""

    def __init__(self):
        self.cosmos_manager = None
        self.cosmos_client = None
        self.database = None
        self.containers = {}
        self.connected = False

    def _display_debug_info(self, message: str, level: str = "info"):
        """Display debug information in Streamlit interface."""
        if level == "error":
            st.error(f"🚨 **Debug Error:** {message}", icon="🚨")
        elif level == "warning":
            st.warning(f"⚠️ **Debug Warning:** {message}", icon="⚠️")
        elif level == "success":
            st.success(f"✅ **Debug Success:** {message}", icon="✅")
        else:
            st.info(f"ℹ️ **Debug Info:** {message}", icon="ℹ️")

        # Also log to Python logger
        getattr(logger, level, logger.info)(message)

    def initialize(self, mode: str = "local") -> bool:
        """Initialize connection to Cosmos DB (query-only, no database management)."""
        try:
            self._display_debug_info(
                f"Starting connection to Cosmos DB with mode: {mode}"
            )

            if not COSMOS_AVAILABLE:
                error_msg = (
                    "Cosmos SDK not available - check if azure-cosmos is installed"
                )
                self._display_debug_info(error_msg, "error")
                return False

            self._display_debug_info("Creating CosmosModeManager...")
            self.cosmos_manager = CosmosModeManager(mode)

            self._display_debug_info("Getting Cosmos client...")
            self.cosmos_client = self.cosmos_manager.get_cosmos_client()

            # Show connection details for debugging
            config = self.cosmos_manager.config
            self._display_debug_info(f"Database config: {config['database']}")
            self._display_debug_info(f"Connection mode: {mode}")

            if mode == "local":
                self._display_debug_info(
                    f"Local endpoint: {config.get('endpoint', 'Not found')}"
                )
            else:
                self._display_debug_info(
                    f"Azure endpoint: {config.get('endpoint', 'Not found')}"
                )

            database_name = self.cosmos_manager.config["database"]
            self._display_debug_info(f"Connecting to database: {database_name}")
            self.database = self.cosmos_client.get_database_client(database_name)

            # Test database connection
            try:
                self.database.read()
                self._display_debug_info(
                    f"Successfully connected to database: {database_name}", "success"
                )
            except CosmosResourceNotFoundError:
                error_msg = f"Database '{database_name}' not found. Please run the ingestion service first to initialize the database."
                self._display_debug_info(error_msg, "error")
                return False
            except Exception as e:
                error_msg = f"Failed to connect to database '{database_name}': {str(e)}"
                self._display_debug_info(error_msg, "error")
                return False

            # Connect to expected containers (no creation, just connection)
            container_names = ["knowledge", "memory", "repositories", "diagrams"]
            self._display_debug_info(
                f"Attempting to connect to containers: {container_names}"
            )

            for container_name in container_names:
                try:
                    self._display_debug_info(
                        f"Connecting to container: {container_name}"
                    )
                    container = self.database.get_container_client(container_name)

                    # Test the connection
                    container.read()
                    self.containers[container_name] = container
                    self._display_debug_info(
                        f"Successfully connected to container: {container_name}",
                        "success",
                    )

                except CosmosResourceNotFoundError:
                    warning_msg = f"Container '{container_name}' not found. This container may not have been created by the ingestion service yet."
                    self._display_debug_info(warning_msg, "warning")
                except Exception as e:
                    error_msg = (
                        f"Error connecting to container '{container_name}': {str(e)}"
                    )
                    self._display_debug_info(error_msg, "error")

            if self.containers:
                self._display_debug_info(
                    f"Successfully connected to {len(self.containers)} containers",
                    "success",
                )
                self.connected = True

                # Display helpful message about missing containers
                missing_containers = set(container_names) - set(self.containers.keys())
                if missing_containers:
                    self._display_debug_info(
                        f"Missing containers: {list(missing_containers)}. These will be created when the ingestion service processes repositories.",
                        "warning",
                    )
            else:
                error_msg = "No containers found. Please run the ingestion service first to initialize the database and process repositories."
                self._display_debug_info(error_msg, "error")
                self.connected = False

            return self.connected

        except Exception as e:
            error_msg = f"Failed to initialize Cosmos DB connection: {str(e)}"
            self._display_debug_info(error_msg, "error")
            st.exception(e)  # Show full exception with traceback
            return False

    def get_container_stats(self) -> Dict[str, int]:
        """Get document count for each container."""
        stats = {}

        if not self.containers:
            self._display_debug_info("No containers available for stats", "warning")
            return stats

        for name, container in self.containers.items():
            try:
                self._display_debug_info(f"Getting stats for container: {name}")
                # Query to count documents
                query = "SELECT VALUE COUNT(1) FROM c"
                items = list(
                    container.query_items(
                        query=query, enable_cross_partition_query=True
                    )
                )
                count = items[0] if items else 0
                stats[name] = count
                self._display_debug_info(f"Container {name}: {count} documents")
            except Exception as e:
                error_msg = f"Error getting stats for {name}: {str(e)}"
                self._display_debug_info(error_msg, "error")
                st.exception(e)
                stats[name] = 0

        return stats

    def get_entities(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get entities from available containers."""
        entities = []

        # Try repositories first for file entities
        if "repositories" in self.containers:
            try:
                query = f"SELECT TOP {limit} * FROM c WHERE c.entity_type = 'file'"
                items = list(
                    self.containers["repositories"].query_items(
                        query=query, enable_cross_partition_query=True
                    )
                )
                entities.extend(items)
                logger.info(f"Loaded {len(items)} file entities from repositories")
            except Exception as e:
                logger.error(f"Error loading file entities: {e}")

        # Try knowledge container for other entities
        if "knowledge" in self.containers and len(entities) < limit:
            try:
                remaining = limit - len(entities)
                query = f"SELECT TOP {remaining} * FROM c"
                items = list(
                    self.containers["knowledge"].query_items(
                        query=query, enable_cross_partition_query=True
                    )
                )
                entities.extend(items)
                logger.info(f"Loaded {len(items)} entities from knowledge")
            except Exception as e:
                logger.error(f"Error loading knowledge entities: {e}")

        return entities

    def get_relationships(self, limit: int = 200) -> List[Dict[str, Any]]:
        """Get relationships from available containers."""
        relationships = []

        # Try repositories for file relationships
        if "repositories" in self.containers:
            try:
                query = f"SELECT TOP {limit} * FROM c WHERE c.relationship_type != null"
                items = list(
                    self.containers["repositories"].query_items(
                        query=query, enable_cross_partition_query=True
                    )
                )
                relationships.extend(items)
                logger.info(f"Loaded {len(items)} relationships from repositories")
            except Exception as e:
                logger.error(f"Error loading relationships: {e}")

        return relationships

    def search_entities(
        self, search_term: str, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Search for entities containing the search term with comprehensive telemetry."""
        st.info(
            f"🔍 **Search Debug:** Searching for '{search_term}' across {len(self.containers)} containers with limit {limit}",
            icon="🔍",
        )
        logger.info(f"Starting search for term: '{search_term}', limit: {limit}")

        results = []
        search_stats = {}

        for container_name, container in self.containers.items():
            try:
                st.info(
                    f"🔍 **Search Debug:** Searching in container '{container_name}'...",
                    icon="🔍",
                )
                logger.debug(f"Searching in container: {container_name}")

                # Search in different fields based on container
                if container_name == "repositories":
                    query = f"""
                    SELECT TOP {limit // len(self.containers)} * FROM c 
                    WHERE CONTAINS(LOWER(c.file_path), LOWER(@search)) 
                       OR CONTAINS(LOWER(c.file_name), LOWER(@search))
                       OR CONTAINS(LOWER(c.content), LOWER(@search))
                    """
                    st.info(
                        "🔍 **Search Debug:** Using repository-specific query for file paths, names, and content",
                        icon="🔍",
                    )
                else:
                    query = f"""
                    SELECT TOP {limit // len(self.containers)} * FROM c 
                    WHERE CONTAINS(LOWER(c.name), LOWER(@search)) 
                       OR CONTAINS(LOWER(c.description), LOWER(@search))
                       OR CONTAINS(LOWER(c.content), LOWER(@search))
                    """
                    st.info(
                        "🔍 **Search Debug:** Using generic query for names, descriptions, and content",
                        icon="🔍",
                    )

                logger.debug(f"Executing query: {query[:100]}...")

                items = list(
                    container.query_items(
                        query=query,
                        parameters=[{"name": "@search", "value": search_term}],
                        enable_cross_partition_query=True,
                    )
                )

                search_stats[container_name] = len(items)
                results.extend(items)

                st.success(
                    f"✅ **Search Debug:** Found {len(items)} results in '{container_name}'",
                    icon="✅",
                )
                logger.info(f"Found {len(items)} items in {container_name}")

            except Exception as e:
                error_msg = f"Error searching in {container_name}: {str(e)}"
                st.error(f"🚨 **Search Debug Error:** {error_msg}", icon="🚨")
                logger.error(error_msg, exc_info=True)
                search_stats[container_name] = 0

        total_results = len(results)
        final_results = results[:limit]

        # Display comprehensive search summary
        st.info(
            f"""
        🔍 **Search Summary:**
        - Search term: '{search_term}'
        - Containers searched: {len(self.containers)}
        - Total results found: {total_results}
        - Results returned: {len(final_results)}
        - Results by container: {search_stats}
        """,
            icon="📊",
        )

        logger.info(
            f"Search completed: term='{search_term}', total_found={total_results}, returned={len(final_results)}, stats={search_stats}"
        )

        return final_results


class SemanticKernelService:
    """Service for AI-powered chat functionality using Semantic Kernel."""

    def __init__(self):
        self.kernel = None
        self.chat_service = None
        self.initialized = False

    def _display_debug_info(self, message: str, level: str = "info"):
        """Display debug information in Streamlit interface."""
        if level == "error":
            st.error(f"🚨 **SK Debug Error:** {message}", icon="🚨")
        elif level == "warning":
            st.warning(f"⚠️ **SK Debug Warning:** {message}", icon="⚠️")
        elif level == "success":
            st.success(f"✅ **SK Debug Success:** {message}", icon="✅")
        else:
            st.info(f"ℹ️ **SK Debug Info:** {message}", icon="ℹ️")

        # Also log to Semantic Kernel logger
        getattr(sk_logger, level, sk_logger.info)(f"SK: {message}")

    def initialize(self) -> bool:
        """Initialize Semantic Kernel with Azure OpenAI and comprehensive telemetry."""
        try:
            self._display_debug_info("🧠 Starting Semantic Kernel initialization...")

            if not SEMANTIC_KERNEL_AVAILABLE:
                error_msg = "Semantic Kernel SDK not available - please install semantic-kernel package"
                self._display_debug_info(error_msg, "error")
                return False

            # Get configuration from environment with detailed logging
            self._display_debug_info(
                "🔍 Checking environment variables for AI service configuration..."
            )

            azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
            azure_openai_deployment_name = os.getenv(
                "AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4"
            )

            openai_api_key = os.getenv("OPENAI_API_KEY")
            openai_model_id = os.getenv("OPENAI_MODEL_ID", "gpt-4")

            # Store config details for debugging (without exposing sensitive data)
            config_details = {
                "azure_endpoint_set": bool(azure_openai_endpoint),
                "azure_key_set": bool(azure_openai_api_key),
                "azure_deployment": azure_openai_deployment_name,
                "openai_key_set": bool(openai_api_key),
                "openai_model": openai_model_id,
            }

            self._display_debug_info(f"Configuration check: {config_details}")

            # Initialize kernel with enhanced logging
            self._display_debug_info("Creating Semantic Kernel instance...")
            self.kernel = sk.Kernel()

            # Set up Semantic Kernel logging
            sk_logger.info("Semantic Kernel instance created successfully")

            # Try Azure OpenAI first, then fallback to OpenAI
            if azure_openai_endpoint and azure_openai_api_key:
                self._display_debug_info(
                    f"🌐 Initializing Azure OpenAI chat service with deployment: {azure_openai_deployment_name}"
                )
                try:
                    self.chat_service = AzureChatCompletion(
                        deployment_name=azure_openai_deployment_name,
                        endpoint=azure_openai_endpoint,
                        api_key=azure_openai_api_key,
                        service_id="azure_chat_completion",
                    )
                    self.kernel.add_service(self.chat_service)
                    self._display_debug_info(
                        "✅ Azure OpenAI service added to kernel", "success"
                    )
                    sk_logger.info(
                        f"Azure OpenAI configured: endpoint={azure_openai_endpoint}, deployment={azure_openai_deployment_name}"
                    )

                except Exception as e:
                    error_msg = f"Failed to initialize Azure OpenAI: {str(e)}"
                    self._display_debug_info(error_msg, "error")
                    st.exception(e)
                    return False

            elif openai_api_key:
                self._display_debug_info(
                    f"🤖 Initializing OpenAI chat service with model: {openai_model_id}"
                )
                try:
                    self.chat_service = OpenAIChatCompletion(
                        ai_model_id=openai_model_id,
                        api_key=openai_api_key,
                        service_id="openai_chat_completion",
                    )
                    self.kernel.add_service(self.chat_service)
                    self._display_debug_info(
                        "✅ OpenAI service added to kernel", "success"
                    )
                    sk_logger.info(f"OpenAI configured: model={openai_model_id}")

                except Exception as e:
                    error_msg = f"Failed to initialize OpenAI: {str(e)}"
                    self._display_debug_info(error_msg, "error")
                    st.exception(e)
                    return False

            else:
                error_msg = "No OpenAI configuration found. Please set AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_API_KEY or OPENAI_API_KEY"
                self._display_debug_info(error_msg, "error")

                # Show configuration help
                st.info(
                    """
                💡 **To enable AI chat, set one of these configurations:**
                
                **For Azure OpenAI:**
                - `AZURE_OPENAI_ENDPOINT` = your Azure OpenAI endpoint
                - `AZURE_OPENAI_API_KEY` = your Azure OpenAI API key  
                - `AZURE_OPENAI_DEPLOYMENT_NAME` = your deployment name (optional, defaults to 'gpt-4')
                
                **For OpenAI:**
                - `OPENAI_API_KEY` = your OpenAI API key
                - `OPENAI_MODEL_ID` = model to use (optional, defaults to 'gpt-4')
                """
                )
                return False

            # Add memory plugin with telemetry
            self._display_debug_info("💾 Adding memory plugin to kernel...")
            try:
                self.kernel.add_plugin(TextMemoryPlugin(NullMemory()), "memory")
                self._display_debug_info(
                    "✅ Memory plugin added successfully", "success"
                )
            except Exception as e:
                warning_msg = f"Failed to add memory plugin (non-critical): {str(e)}"
                self._display_debug_info(warning_msg, "warning")

            self.initialized = True
            success_msg = f"✅ Semantic Kernel initialized successfully with {type(self.chat_service).__name__}"
            self._display_debug_info(success_msg, "success")
            sk_logger.info("Semantic Kernel initialization completed successfully")

            return True

        except Exception as e:
            error_msg = f"Failed to initialize Semantic Kernel: {str(e)}"
            self._display_debug_info(error_msg, "error")
            sk_logger.error(error_msg, exc_info=True)
            st.exception(e)
            return False

    async def chat(self, message: str, context: str = "") -> str:
        """Process a chat message with optional context and comprehensive telemetry."""
        try:
            self._display_debug_info(f"💬 Processing chat message: {message[:50]}...")

            if not self.initialized:
                error_msg = "❌ Chat service not available. Please check your OpenAI configuration."
                self._display_debug_info(error_msg, "error")
                return error_msg

            # Create enhanced prompt with context
            system_prompt = """You are an AI assistant for the Mosaic Knowledge Graph system. 
            You help users understand and navigate code repositories, relationships, and system architecture.
            
            You have access to repository data including files, dependencies, and code relationships.
            Provide helpful, accurate answers about the codebase structure and functionality.
            
            When discussing code, be specific about file paths, functions, and relationships when possible.
            """

            if context:
                system_prompt += f"\n\nCurrent Context:\n{context}"
                self._display_debug_info(f"Added context: {len(context)} characters")

            user_prompt = f"""
            System: {system_prompt}
            
            User: {message}
            
            Assistant: """

            self._display_debug_info("🚀 Invoking Semantic Kernel prompt...")
            sk_logger.debug(f"Prompt length: {len(user_prompt)} characters")

            # Execute the prompt using Semantic Kernel with telemetry
            result = await self.kernel.invoke_prompt(
                function_name="chat_response", plugin_name="chat", prompt=user_prompt
            )

            response = str(result)
            self._display_debug_info(
                f"✅ Received response: {len(response)} characters", "success"
            )
            sk_logger.info(
                f"Chat completion successful: input={len(message)} chars, output={len(response)} chars"
            )

            return response

        except Exception as e:
            error_msg = f"Error in chat processing: {str(e)}"
            self._display_debug_info(error_msg, "error")
            sk_logger.error(error_msg, exc_info=True)
            st.exception(e)
            return f"❌ Error processing your request: {str(e)}"


def create_graph_visualization(entities: List[Dict], relationships: List[Dict]):
    """Create an interactive graph visualization using streamlit-agraph."""

    nodes = []
    edges = []

    # Dark mode optimized color mapping for different entity types
    color_map = {
        "file": "#3FB950",  # Success green for files
        "class": "#58A6FF",  # Primary blue for classes
        "function": "#D29922",  # Warning amber for functions
        "module": "#BC8CFF",  # Purple for modules (desaturated)
        "dependency": "#F85149",  # Error red for dependencies
        "default": "#8B949E",  # Muted grey for unknown
    }

    # Create nodes from entities
    entity_ids = set()
    for entity in entities:
        entity_id = entity.get("id", entity.get("file_path", str(uuid.uuid4())))
        if entity_id in entity_ids:
            continue
        entity_ids.add(entity_id)

        # Determine entity type and color
        entity_type = entity.get("entity_type", entity.get("type", "default"))
        color = color_map.get(entity_type, color_map["default"])

        # Create label
        label = entity.get(
            "name", entity.get("file_name", entity.get("file_path", "Unknown"))
        )
        if len(label) > 30:
            label = label[:27] + "..."

        # Calculate size based on lines or complexity
        size = 20
        if "lines_of_code" in entity:
            size = min(50, max(15, entity["lines_of_code"] // 10))
        elif "lines" in entity:
            size = min(50, max(15, entity["lines"] // 10))
        elif "complexity" in entity:
            size = min(50, max(15, entity["complexity"] * 3))

        node = Node(
            id=entity_id,
            label=label,
            size=size,
            color=color,
            title=f"Type: {entity_type}\nPath: {entity.get('file_path', 'N/A')}\nLines: {entity.get('lines_of_code', entity.get('lines', 'N/A'))}",
        )
        nodes.append(node)

    # Create edges from relationships
    for relationship in relationships:
        source = relationship.get(
            "source_id", relationship.get("from", relationship.get("source"))
        )
        target = relationship.get(
            "target_id", relationship.get("to", relationship.get("target"))
        )

        if source and target and source in entity_ids and target in entity_ids:
            rel_type = relationship.get(
                "relationship_type", relationship.get("type", "related")
            )
            edge = Edge(
                source=source,
                target=target,
                label=rel_type,
                color="#8B949E",  # Dark mode muted grey for edges
                title=f"Relationship: {rel_type}",
            )
            edges.append(edge)

    # Configure the graph
    config = Config(
        width="100%",
        height=600,
        directed=True,
        physics=True,
        hierarchical=False,
        node_spacing=200,
        spring_length=100,
        spring_strength=0.05,
        damping=0.1,
    )

    return nodes, edges, config


def main():
    """Main Streamlit application with comprehensive telemetry and debugging."""

    # Display debug information in the main interface
    st.info("**App Debug:** Starting Mosaic Knowledge Graph application...", icon="🚀")
    logger.info("Starting Mosaic Knowledge Graph application")

    # Header
    st.markdown(
        """
        <div class="main-header">
            <h1>🎯 Mosaic Knowledge Graph</h1>
            <p>Interactive repository exploration and OmniRAG visualization</p>
            <small>Query-only UI - Database management handled by Mosaic Ingestion Service</small>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Initialize session state with debug information
    st.info("**App Debug:** Initializing session state and services...", icon="🔧")

    if "data_service" not in st.session_state:
        st.session_state.data_service = MosaicDataService()
        st.info("**App Debug:** Created new MosaicDataService instance", icon="🔧")
        logger.info("Created new MosaicDataService instance")

    if "sk_service" not in st.session_state:
        st.session_state.sk_service = SemanticKernelService()
        st.info("**App Debug:** Created new SemanticKernelService instance", icon="🔧")
        logger.info("Created new SemanticKernelService instance")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        st.info("**App Debug:** Initialized empty chat history", icon="🔧")

    if "current_entities" not in st.session_state:
        st.session_state.current_entities = []
        st.info("**App Debug:** Initialized empty entities list", icon="🔧")

    if "current_relationships" not in st.session_state:
        st.session_state.current_relationships = []
        st.info("**App Debug:** Initialized empty relationships list", icon="🔧")

    # Display current session state
    st.info(
        f"""
    **Session State Summary:**
    - Data Service Connected: {st.session_state.data_service.connected}
    - AI Service Initialized: {st.session_state.sk_service.initialized}
    - Chat History Length: {len(st.session_state.chat_history)}
    - Current Entities: {len(st.session_state.current_entities)}
    - Current Relationships: {len(st.session_state.current_relationships)}
    """,
        icon="📊",
    )

    logger.info(
        f"Session state initialized: data_connected={st.session_state.data_service.connected}, "
        f"ai_initialized={st.session_state.sk_service.initialized}, "
        f"chat_history={len(st.session_state.chat_history)}, "
        f"entities={len(st.session_state.current_entities)}, "
        f"relationships={len(st.session_state.current_relationships)}"
    )

    # Sidebar for configuration and controls
    with st.sidebar:
        st.title("🔧 Configuration")

        # Environment variables debug section
        with st.expander("🔍 Environment Variables Debug", expanded=False):
            env_status = {
                "COSMOS_MODE": os.getenv("COSMOS_MODE", "Not Set"),
                "AZURE_OPENAI_ENDPOINT": (
                    "Set" if os.getenv("AZURE_OPENAI_ENDPOINT") else "Not Set"
                ),
                "AZURE_OPENAI_API_KEY": (
                    "Set" if os.getenv("AZURE_OPENAI_API_KEY") else "Not Set"
                ),
                "AZURE_OPENAI_DEPLOYMENT_NAME": os.getenv(
                    "AZURE_OPENAI_DEPLOYMENT_NAME", "Not Set"
                ),
                "OPENAI_API_KEY": "Set" if os.getenv("OPENAI_API_KEY") else "Not Set",
                "OPENAI_MODEL_ID": os.getenv("OPENAI_MODEL_ID", "Not Set"),
                "SEMANTICKERNEL_EXPERIMENTAL_GENAI_ENABLE_OTEL_DIAGNOSTICS": os.getenv(
                    "SEMANTICKERNEL_EXPERIMENTAL_GENAI_ENABLE_OTEL_DIAGNOSTICS",
                    "Not Set",
                ),
                "SEMANTICKERNEL_EXPERIMENTAL_GENAI_ENABLE_OTEL_DIAGNOSTICS_SENSITIVE": os.getenv(
                    "SEMANTICKERNEL_EXPERIMENTAL_GENAI_ENABLE_OTEL_DIAGNOSTICS_SENSITIVE",
                    "Not Set",
                ),
            }

            for key, value in env_status.items():
                if "Not Set" in value:
                    st.error(f"❌ {key}: {value}")
                else:
                    st.success(f"✅ {key}: {value}")

        # Mode selection
        mode = st.selectbox(
            "Database Mode",
            ["local", "azure"],
            index=0,
            help="Select whether to connect to local Cosmos DB emulator or Azure Cosmos DB",
        )

        st.info(f"**Config Debug:** Selected database mode: {mode}", icon="🔧")

        # Connection status with enhanced debugging
        if st.button("🔌 Connect to Database"):
            st.info(
                f"🔌 **Connection Debug:** Attempting to connect in {mode} mode...",
                icon="🔌",
            )
            logger.info(f"Attempting database connection in {mode} mode")

            with st.spinner("Connecting to Cosmos DB..."):
                try:
                    if st.session_state.data_service.initialize(mode):
                        st.success("✅ Connected to Cosmos DB successfully!", icon="✅")
                        logger.info(
                            f"Successfully connected to Cosmos DB in {mode} mode"
                        )
                    else:
                        st.error("❌ Failed to connect to Cosmos DB", icon="❌")
                        st.info(
                            """
                        💡 **Connection Failed?**
                        
                        The Mosaic UI is a query-only tool. If connection fails:
                        
                        1. **For Local Mode**: Ensure the Cosmos DB emulator is running
                        2. **For Azure Mode**: Verify your Azure credentials and connection
                        3. **Missing Database/Containers**: Run the **Mosaic Ingestion Service** first
                        
                        The Ingestion Service handles:
                        - Database and container creation
                        - Repository processing and analysis
                        - OmniRAG data ingestion
                        
                        The UI only queries existing data.
                        """,
                            icon="💡",
                        )
                        logger.error(f"Failed to connect to Cosmos DB in {mode} mode")
                except Exception as e:
                    error_msg = f"Exception during database connection: {str(e)}"
                    st.error(f"🚨 **Connection Error:** {error_msg}", icon="🚨")
                    logger.error(error_msg, exc_info=True)
                    st.exception(e)

        # Display connection status
        if st.session_state.data_service.connected:
            st.markdown(
                '<div class="data-status">🟢 Database Connected</div>',
                unsafe_allow_html=True,
            )

            # Show container stats
            stats = st.session_state.data_service.get_container_stats()
            st.subheader("📊 Container Stats")
            for container, count in stats.items():
                st.metric(container, count)
        else:
            st.markdown(
                '<div class="error-status">🔴 Database Disconnected</div>',
                unsafe_allow_html=True,
            )

        # AI Configuration
        st.title("🤖 AI Configuration")

        # Display current AI service status
        st.info(
            f"🤖 **AI Debug:** Current AI service status: {'Initialized' if st.session_state.sk_service.initialized else 'Not Initialized'}",
            icon="🤖",
        )

        # Initialize Semantic Kernel with enhanced debugging
        if st.button("🧠 Initialize AI Chat"):
            st.info(
                "🧠 **AI Debug:** Starting Semantic Kernel initialization...", icon="🧠"
            )
            logger.info("Starting Semantic Kernel initialization")

            with st.spinner("Initializing Semantic Kernel..."):
                try:
                    if st.session_state.sk_service.initialize():
                        st.success(
                            "✅ AI Chat Ready - Semantic Kernel initialized successfully!",
                            icon="✅",
                        )
                        logger.info(
                            "Semantic Kernel initialization completed successfully"
                        )
                    else:
                        st.error(
                            "❌ AI Chat Unavailable - Semantic Kernel initialization failed",
                            icon="❌",
                        )
                        logger.error("Semantic Kernel initialization failed")
                except Exception as e:
                    error_msg = f"Exception during AI initialization: {str(e)}"
                    st.error(f"🚨 **AI Error:** {error_msg}", icon="🚨")
                    logger.error(error_msg, exc_info=True)
                    st.exception(e)

        # Show AI status with detailed information
        if st.session_state.sk_service.initialized:
            st.markdown(
                '<div class="data-status">🟢 AI Chat Ready</div>',
                unsafe_allow_html=True,
            )

            # Display AI service configuration details if available
            if (
                hasattr(st.session_state.sk_service, "config_details")
                and st.session_state.sk_service.config_details
            ):
                with st.expander("🤖 AI Service Details", expanded=False):
                    for (
                        key,
                        value,
                    ) in st.session_state.sk_service.config_details.items():
                        if value:
                            st.success(f"✅ {key}: {value}")
                        else:
                            st.error(f"❌ {key}: {value}")

        else:
            st.markdown(
                '<div class="error-status">🔴 AI Chat Unavailable</div>',
                unsafe_allow_html=True,
            )
            st.info(
                """
            💡 **To enable AI chat:**
            1. Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY for Azure OpenAI
            2. Or set OPENAI_API_KEY for OpenAI
            3. Click 'Initialize AI Chat' button
            
            📊 **For full telemetry, also set:**
            - SEMANTICKERNEL_EXPERIMENTAL_GENAI_ENABLE_OTEL_DIAGNOSTICS=true
            - SEMANTICKERNEL_EXPERIMENTAL_GENAI_ENABLE_OTEL_DIAGNOSTICS_SENSITIVE=true
            """
            )

        # Logging and Telemetry Section
        with st.expander("📊 Logging & Telemetry Settings", expanded=False):
            st.write("**Current Logging Configuration:**")
            st.info(f"Python Logger Level: {logger.level}")
            st.info(f"Semantic Kernel Logger Level: {sk_logger.level}")
            st.info(
                "Streamlit Error Details: Full (configured in .streamlit/config.toml)"
            )

            st.write("**OpenTelemetry Status:**")
            otel_diag = (
                os.getenv(
                    "SEMANTICKERNEL_EXPERIMENTAL_GENAI_ENABLE_OTEL_DIAGNOSTICS", "false"
                ).lower()
                == "true"
            )
            otel_sensitive = (
                os.getenv(
                    "SEMANTICKERNEL_EXPERIMENTAL_GENAI_ENABLE_OTEL_DIAGNOSTICS_SENSITIVE",
                    "false",
                ).lower()
                == "true"
            )

            if otel_diag:
                st.success("✅ Semantic Kernel OpenTelemetry Diagnostics: Enabled")
            else:
                st.warning("⚠️ Semantic Kernel OpenTelemetry Diagnostics: Disabled")

            if otel_sensitive:
                st.success("✅ Semantic Kernel Sensitive Data Logging: Enabled")
            else:
                st.warning("⚠️ Semantic Kernel Sensitive Data Logging: Disabled")

    # Main content area with tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Dashboard", "🕸️ Graph View", "💬 AI Chat", "🔍 Search"]
    )

    with tab1:
        st.header("📊 Repository Dashboard")

        if not st.session_state.data_service.connected:
            st.warning(
                """
            ⚠️ **Database Not Connected**
            
            Please connect to the database first using the sidebar.
            
            **Note**: This UI only queries existing data. If no database exists:
            1. Run the **Mosaic Ingestion Service** first to process repositories
            2. The Ingestion Service will create the database and containers
            3. Then return here to explore the processed data
            """
            )
            return

        # Load and display data overview
        col1, col2 = st.columns(2)

        with col1:
            if st.button("📥 Load Repository Data"):
                with st.spinner("Loading entities and relationships..."):
                    entities = st.session_state.data_service.get_entities(100)
                    relationships = st.session_state.data_service.get_relationships(200)

                    st.session_state.current_entities = entities
                    st.session_state.current_relationships = relationships

                    st.success(
                        f"✅ Loaded {len(entities)} entities and {len(relationships)} relationships"
                    )

        with col2:
            if st.session_state.current_entities:
                st.metric("Entities Loaded", len(st.session_state.current_entities))
                st.metric(
                    "Relationships Loaded", len(st.session_state.current_relationships)
                )

        # Display entity overview
        if st.session_state.current_entities:
            st.subheader("📁 Repository Files")

            # Create DataFrame for display
            entity_data = []
            for entity in st.session_state.current_entities[:20]:  # Show first 20
                entity_data.append(
                    {
                        "File": entity.get("file_name", entity.get("name", "Unknown")),
                        "Path": entity.get("file_path", "N/A"),
                        "Type": entity.get("entity_type", entity.get("type", "file")),
                        "Lines": entity.get(
                            "lines_of_code", entity.get("lines", "N/A")
                        ),
                        "Size": entity.get("file_size", "N/A"),
                    }
                )

            df = pd.DataFrame(entity_data)
            st.dataframe(df, use_container_width=True)

            # File type distribution
            if len(entity_data) > 0:
                type_counts = df["Type"].value_counts()
                fig = px.pie(
                    values=type_counts.values,
                    names=type_counts.index,
                    title="Entity Type Distribution",
                )
                st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.header("🕸️ Interactive Graph Visualization")

        if not st.session_state.current_entities:
            st.warning("Please load repository data from the Dashboard tab first.")
            return

        # Graph controls
        st.markdown('<div class="graph-controls">', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            max_nodes = st.slider("Max Nodes", 10, 100, 50)
        with col2:
            max_edges = st.slider("Max Edges", 10, 200, 100)
        with col3:
            physics_enabled = st.checkbox("Physics", True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Create and display graph
        entities_subset = st.session_state.current_entities[:max_nodes]
        relationships_subset = st.session_state.current_relationships[:max_edges]

        nodes, edges, config = create_graph_visualization(
            entities_subset, relationships_subset
        )
        config.physics = physics_enabled

        if nodes:
            st.subheader(f"Graph: {len(nodes)} nodes, {len(edges)} edges")
            selected_node = agraph(nodes=nodes, edges=edges, config=config)

            # Show selected node details
            if selected_node:
                st.subheader("🔍 Selected Node Details")
                for entity in entities_subset:
                    entity_id = entity.get("id", entity.get("file_path"))
                    if entity_id == selected_node:
                        st.json(entity)
                        break
        else:
            st.info(
                "No graph data available. Please ensure entities are loaded correctly."
            )

    with tab3:
        st.header("💬 AI-Powered Repository Chat")

        if not st.session_state.sk_service.initialized:
            st.warning("Please initialize the AI chat service from the sidebar first.")
            return

        # Display chat history
        st.subheader("Chat History")
        chat_container = st.container()

        with chat_container:
            for i, message in enumerate(st.session_state.chat_history):
                if message["role"] == "user":
                    st.markdown(
                        f'<div class="chat-message user-message">👤 **You:** {message["content"]}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div class="chat-message assistant-message">🤖 **Assistant:** {message["content"]}</div>',
                        unsafe_allow_html=True,
                    )

        # Chat input
        user_input = st.text_input(
            "Ask about the repository:",
            placeholder="e.g., What are the main components of this system?",
        )

        col1, col2 = st.columns([1, 4])
        with col1:
            send_button = st.button("Send", type="primary")
        with col2:
            clear_button = st.button("Clear History")

        if clear_button:
            st.session_state.chat_history = []
            st.rerun()

        if send_button and user_input:
            # Add user message to history
            st.session_state.chat_history.append(
                {"role": "user", "content": user_input}
            )

            # Create context from current entities
            context = ""
            if st.session_state.current_entities:
                context = f"Repository contains {len(st.session_state.current_entities)} files/entities and {len(st.session_state.current_relationships)} relationships."

                # Add sample entity info
                sample_entities = st.session_state.current_entities[:5]
                context += " Key files include: "
                context += ", ".join(
                    [
                        e.get("file_path", e.get("name", "Unknown"))
                        for e in sample_entities
                    ]
                )

            # Get AI response
            with st.spinner("🤖 Thinking..."):

                async def get_response():
                    return await st.session_state.sk_service.chat(user_input, context)

                # Run async function
                response = asyncio.run(get_response())

            # Add assistant response to history
            st.session_state.chat_history.append(
                {"role": "assistant", "content": response}
            )

            # Rerun to show new messages
            st.rerun()

    with tab4:
        st.header("🔍 Search Repository")

        if not st.session_state.data_service.connected:
            st.warning("Please connect to the database first using the sidebar.")
            return

        # Search interface
        search_term = st.text_input(
            "Search for files, functions, or content:",
            placeholder="e.g., main.py, function, class",
        )

        if st.button("🔍 Search") and search_term:
            with st.spinner("Searching repository..."):
                results = st.session_state.data_service.search_entities(search_term, 50)

            if results:
                st.success(f"Found {len(results)} results for '{search_term}'")

                # Display results
                for i, result in enumerate(results):
                    with st.expander(
                        f"📄 {result.get('file_name', result.get('name', 'Unknown'))} - {result.get('file_path', 'N/A')}"
                    ):
                        col1, col2 = st.columns(2)

                        with col1:
                            st.write("**Type:**", result.get("entity_type", "Unknown"))
                            st.write("**Path:**", result.get("file_path", "N/A"))
                            st.write("**Size:**", result.get("file_size", "N/A"))

                        with col2:
                            st.write(
                                "**Lines:**",
                                result.get("lines_of_code", result.get("lines", "N/A")),
                            )
                            st.write(
                                "**Last Modified:**", result.get("last_modified", "N/A")
                            )

                        # Show content preview if available
                        content = result.get("content", result.get("summary", ""))
                        if content:
                            st.write("**Content Preview:**")
                            st.text(
                                content[:500] + "..." if len(content) > 500 else content
                            )
            else:
                st.info(
                    f"No results found for '{search_term}'. Try different keywords."
                )


if __name__ == "__main__":
    main()
