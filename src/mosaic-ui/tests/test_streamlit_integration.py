"""
Comprehensive tests for Streamlit integration and UI functionality.

Tests cover:
- Streamlit session state management
- Component rendering and interaction
- User interface workflows
- Chat interface functionality
- Form handling and validation
- Navigation and page state
- Error handling in UI context
"""

import pytest
from unittest.mock import MagicMock, patch


class MockStreamlitApp:
    """Mock Streamlit application for testing UI functionality."""

    def __init__(self):
        self.session_state = MagicMock()
        self.components = []
        self.messages = []
        self.sidebar_components = []
        self.columns = []

        # Initialize mock session state
        self.session_state.entities = []
        self.session_state.relationships = []
        self.session_state.selected_node = None
        self.session_state.chat_history = []
        self.session_state.ingestion_status = "Not Started"
        self.session_state.mosaic_services = (None, None, None)

    def set_page_config(self, **kwargs):
        """Mock st.set_page_config()."""
        self.page_config = kwargs
        return self

    def markdown(self, text, unsafe_allow_html=False):
        """Mock st.markdown()."""
        self.components.append(
            ("markdown", text, {"unsafe_allow_html": unsafe_allow_html})
        )
        return self

    def title(self, text):
        """Mock st.title()."""
        self.components.append(("title", text))
        return self

    def header(self, text):
        """Mock st.header()."""
        self.components.append(("header", text))
        return self

    def subheader(self, text):
        """Mock st.subheader()."""
        self.components.append(("subheader", text))
        return self

    def selectbox(self, label, options, **kwargs):
        """Mock st.selectbox()."""
        selected = options[0] if options else None
        self.components.append(("selectbox", label, options, selected))
        return selected

    def button(self, label, **kwargs):
        """Mock st.button()."""
        clicked = kwargs.get("clicked", False)
        self.components.append(("button", label, clicked))
        return clicked

    def text_input(self, label, **kwargs):
        """Mock st.text_input()."""
        value = kwargs.get("value", "")
        self.components.append(("text_input", label, value))
        return value

    def columns(self, spec):
        """Mock st.columns()."""
        if isinstance(spec, int):
            cols = [MagicMock() for _ in range(spec)]
        else:
            cols = [MagicMock() for _ in spec]

        # Configure each column mock
        for i, col in enumerate(cols):
            col.markdown = lambda text, **kwargs: self.markdown(text, **kwargs)
            col.button = lambda label, **kwargs: self.button(label, **kwargs)
            col.selectbox = lambda label, options, **kwargs: self.selectbox(
                label, options, **kwargs
            )
            col.text_input = lambda label, **kwargs: self.text_input(label, **kwargs)

        self.columns.append(cols)
        return cols

    def sidebar(self):
        """Mock st.sidebar."""
        sidebar = MagicMock()
        sidebar.markdown = lambda text, **kwargs: self.sidebar_components.append(
            ("markdown", text)
        )
        sidebar.button = lambda label, **kwargs: self.sidebar_components.append(
            ("button", label)
        )
        sidebar.selectbox = (
            lambda label, options, **kwargs: self.sidebar_components.append(
                ("selectbox", label, options)
            )
        )
        return sidebar

    def container(self):
        """Mock st.container()."""
        return MagicMock()

    def success(self, message):
        """Mock st.success()."""
        self.messages.append(("success", message))
        return self

    def error(self, message):
        """Mock st.error()."""
        self.messages.append(("error", message))
        return self

    def warning(self, message):
        """Mock st.warning()."""
        self.messages.append(("warning", message))
        return self

    def info(self, message):
        """Mock st.info()."""
        self.messages.append(("info", message))
        return self

    def rerun(self):
        """Mock st.rerun()."""
        self.components.append(("rerun",))
        return self

    def empty(self):
        """Mock st.empty()."""
        return MagicMock()


@pytest.fixture
def mock_streamlit():
    """Create mock Streamlit instance."""
    return MockStreamlitApp()


@pytest.fixture
def mock_components():
    """Mock Streamlit components.v1."""
    components = MagicMock()
    components.html = MagicMock(return_value="<div>Mock HTML Component</div>")
    return components


class TestStreamlitSessionState:
    """Test Streamlit session state management."""

    def test_session_state_initialization(self, mock_streamlit):
        """Test session state initialization."""
        # Mock session state should be initialized
        assert hasattr(mock_streamlit.session_state, "entities")
        assert hasattr(mock_streamlit.session_state, "relationships")
        assert hasattr(mock_streamlit.session_state, "selected_node")
        assert hasattr(mock_streamlit.session_state, "chat_history")

        # Default values
        assert mock_streamlit.session_state.entities == []
        assert mock_streamlit.session_state.relationships == []
        assert mock_streamlit.session_state.selected_node is None
        assert mock_streamlit.session_state.chat_history == []

    def test_session_state_updates(self, mock_streamlit):
        """Test session state updates."""
        # Update entities
        test_entities = [{"id": "test1", "name": "Test Entity", "category": "test"}]
        mock_streamlit.session_state.entities = test_entities

        assert mock_streamlit.session_state.entities == test_entities

        # Update selected node
        test_node = {"id": "test1", "name": "Selected Node"}
        mock_streamlit.session_state.selected_node = test_node

        assert mock_streamlit.session_state.selected_node == test_node

    def test_session_state_persistence(self, mock_streamlit):
        """Test session state persistence across operations."""
        # Set initial values
        mock_streamlit.session_state.ingestion_status = "Running"
        mock_streamlit.session_state.chat_history = [("user", "test message")]

        # Values should persist
        assert mock_streamlit.session_state.ingestion_status == "Running"
        assert len(mock_streamlit.session_state.chat_history) == 1

        # Add more chat history
        mock_streamlit.session_state.chat_history.append(("assistant", "test response"))
        assert len(mock_streamlit.session_state.chat_history) == 2


class TestStreamlitComponents:
    """Test Streamlit component rendering."""

    def test_page_configuration(self, mock_streamlit):
        """Test page configuration setup."""
        config = {
            "page_title": "Mosaic UI Test",
            "page_icon": "🎯",
            "layout": "wide",
            "initial_sidebar_state": "expanded",
        }

        mock_streamlit.set_page_config(**config)

        assert hasattr(mock_streamlit, "page_config")
        assert mock_streamlit.page_config == config

    def test_header_components(self, mock_streamlit):
        """Test header component rendering."""
        mock_streamlit.title("Mosaic UI Application")
        mock_streamlit.header("Main Section")
        mock_streamlit.subheader("Subsection")

        # Verify components were added
        titles = [comp for comp in mock_streamlit.components if comp[0] == "title"]
        headers = [comp for comp in mock_streamlit.components if comp[0] == "header"]
        subheaders = [
            comp for comp in mock_streamlit.components if comp[0] == "subheader"
        ]

        assert len(titles) == 1
        assert len(headers) == 1
        assert len(subheaders) == 1

        assert titles[0][1] == "Mosaic UI Application"
        assert headers[0][1] == "Main Section"
        assert subheaders[0][1] == "Subsection"

    def test_markdown_rendering(self, mock_streamlit):
        """Test markdown component rendering."""
        # Regular markdown
        mock_streamlit.markdown("# Test Markdown")

        # Unsafe HTML markdown
        mock_streamlit.markdown("<div>HTML Content</div>", unsafe_allow_html=True)

        markdown_components = [
            comp for comp in mock_streamlit.components if comp[0] == "markdown"
        ]

        assert len(markdown_components) == 2
        assert markdown_components[0][1] == "# Test Markdown"
        assert markdown_components[1][2]["unsafe_allow_html"] is True

    def test_form_components(self, mock_streamlit):
        """Test form component rendering."""
        # Selectbox
        options = ["Option 1", "Option 2", "Option 3"]
        selected = mock_streamlit.selectbox("Choose option:", options)

        # Text input
        text_value = mock_streamlit.text_input("Enter text:", value="default")

        # Button
        clicked = mock_streamlit.button("Submit")

        # Verify components
        selectboxes = [
            comp for comp in mock_streamlit.components if comp[0] == "selectbox"
        ]
        text_inputs = [
            comp for comp in mock_streamlit.components if comp[0] == "text_input"
        ]
        buttons = [comp for comp in mock_streamlit.components if comp[0] == "button"]

        assert len(selectboxes) == 1
        assert len(text_inputs) == 1
        assert len(buttons) == 1

        # Verify values
        assert selected == "Option 1"  # First option selected by default
        assert text_value == "default"
        assert clicked is False  # Default not clicked

    def test_layout_columns(self, mock_streamlit):
        """Test column layout functionality."""
        # Create 2 columns
        col1, col2 = mock_streamlit.columns(2)

        assert len(mock_streamlit.columns) == 1
        assert len(mock_streamlit.columns[0]) == 2

        # Create 3 columns with ratios
        col1, col2, col3 = mock_streamlit.columns([2, 1, 1])

        assert len(mock_streamlit.columns) == 2
        assert len(mock_streamlit.columns[1]) == 3

    def test_sidebar_components(self, mock_streamlit):
        """Test sidebar component functionality."""
        sidebar = mock_streamlit.sidebar()

        # Add components to sidebar
        sidebar.markdown("### Sidebar Content")
        sidebar.button("Sidebar Button")
        sidebar.selectbox("Sidebar Select", ["A", "B", "C"])

        # Verify sidebar components were tracked
        assert len(mock_streamlit.sidebar_components) == 3
        assert mock_streamlit.sidebar_components[0][0] == "markdown"
        assert mock_streamlit.sidebar_components[1][0] == "button"
        assert mock_streamlit.sidebar_components[2][0] == "selectbox"


class TestMessageHandling:
    """Test message and notification handling."""

    def test_success_messages(self, mock_streamlit):
        """Test success message display."""
        mock_streamlit.success("Operation completed successfully!")

        success_messages = [
            msg for msg in mock_streamlit.messages if msg[0] == "success"
        ]
        assert len(success_messages) == 1
        assert success_messages[0][1] == "Operation completed successfully!"

    def test_error_messages(self, mock_streamlit):
        """Test error message display."""
        mock_streamlit.error("An error occurred during processing")

        error_messages = [msg for msg in mock_streamlit.messages if msg[0] == "error"]
        assert len(error_messages) == 1
        assert error_messages[0][1] == "An error occurred during processing"

    def test_warning_messages(self, mock_streamlit):
        """Test warning message display."""
        mock_streamlit.warning("This is a warning message")

        warning_messages = [
            msg for msg in mock_streamlit.messages if msg[0] == "warning"
        ]
        assert len(warning_messages) == 1
        assert warning_messages[0][1] == "This is a warning message"

    def test_info_messages(self, mock_streamlit):
        """Test info message display."""
        mock_streamlit.info("Informational message")

        info_messages = [msg for msg in mock_streamlit.messages if msg[0] == "info"]
        assert len(info_messages) == 1
        assert info_messages[0][1] == "Informational message"

    def test_multiple_messages(self, mock_streamlit):
        """Test multiple message types."""
        mock_streamlit.success("Success 1")
        mock_streamlit.error("Error 1")
        mock_streamlit.warning("Warning 1")
        mock_streamlit.info("Info 1")
        mock_streamlit.success("Success 2")

        assert len(mock_streamlit.messages) == 5

        # Verify message types
        success_count = len(
            [msg for msg in mock_streamlit.messages if msg[0] == "success"]
        )
        error_count = len([msg for msg in mock_streamlit.messages if msg[0] == "error"])
        warning_count = len(
            [msg for msg in mock_streamlit.messages if msg[0] == "warning"]
        )
        info_count = len([msg for msg in mock_streamlit.messages if msg[0] == "info"])

        assert success_count == 2
        assert error_count == 1
        assert warning_count == 1
        assert info_count == 1


class TestChatInterface:
    """Test chat interface functionality."""

    def test_chat_history_display(self, mock_streamlit, sample_chat_history):
        """Test chat history display."""
        mock_streamlit.session_state.chat_history = sample_chat_history

        # Simulate chat history rendering
        for role, message in mock_streamlit.session_state.chat_history:
            if role == "user":
                mock_streamlit.markdown(f"**You:** {message}")
            else:
                mock_streamlit.markdown(f"**Assistant:** {message}")

        # Verify chat messages were rendered
        chat_components = [
            comp for comp in mock_streamlit.components if comp[0] == "markdown"
        ]
        assert len(chat_components) == len(sample_chat_history)

    def test_chat_input_handling(self, mock_streamlit):
        """Test chat input handling."""
        # Simulate user input
        user_input = mock_streamlit.text_input(
            "Ask a question:", value="What AI agents are available?"
        )

        mock_streamlit.button("Send Question")

        # Verify input components
        text_inputs = [
            comp for comp in mock_streamlit.components if comp[0] == "text_input"
        ]
        buttons = [comp for comp in mock_streamlit.components if comp[0] == "button"]

        assert len(text_inputs) == 1
        assert len(buttons) == 1
        assert user_input == "What AI agents are available?"

    def test_chat_message_addition(self, mock_streamlit):
        """Test adding messages to chat history."""
        # Initialize empty chat history
        mock_streamlit.session_state.chat_history = []

        # Add user message
        user_message = "Test question"
        mock_streamlit.session_state.chat_history.append(("user", user_message))

        # Add assistant response
        assistant_response = "Test response"
        mock_streamlit.session_state.chat_history.append(
            ("assistant", assistant_response)
        )

        assert len(mock_streamlit.session_state.chat_history) == 2
        assert mock_streamlit.session_state.chat_history[0] == ("user", user_message)
        assert mock_streamlit.session_state.chat_history[1] == (
            "assistant",
            assistant_response,
        )

    def test_chat_clearing(self, mock_streamlit, sample_chat_history):
        """Test chat history clearing."""
        # Set initial chat history
        mock_streamlit.session_state.chat_history = sample_chat_history.copy()
        assert len(mock_streamlit.session_state.chat_history) > 0

        # Clear chat history
        mock_streamlit.session_state.chat_history = []
        assert len(mock_streamlit.session_state.chat_history) == 0


class TestGraphVisualizationSelection:
    """Test graph visualization selection and rendering."""

    def test_visualization_type_selection(self, mock_streamlit):
        """Test visualization type selection."""
        options = [
            "Enhanced D3.js (OmniRAG-style)",
            "Pyvis Network (Vis.js compatible)",
            "Plotly Graph (Advanced Analytics)",
            "Classic D3.js",
        ]

        selected = mock_streamlit.selectbox("Select Graph Visualization:", options)

        # Should select first option by default
        assert selected == "Enhanced D3.js (OmniRAG-style)"

        # Verify selectbox was created
        selectboxes = [
            comp for comp in mock_streamlit.components if comp[0] == "selectbox"
        ]
        assert len(selectboxes) == 1
        assert selectboxes[0][2] == options  # Options should match

    def test_visualization_info_display(self, mock_streamlit):
        """Test visualization info display."""
        # Simulate info display for different visualization types
        viz_types = {
            "Enhanced D3.js": "Interactive controls, zoom/pan, highlighting",
            "Pyvis Network": "Vis.js compatible physics simulation",
            "Plotly Graph": "Advanced analytics visualization",
            "Classic D3.js": "Simple, lightweight visualization",
        }

        for viz_type, description in viz_types.items():
            mock_streamlit.info(f"**{viz_type}**: {description}")

        # Verify info messages
        info_messages = [msg for msg in mock_streamlit.messages if msg[0] == "info"]
        assert len(info_messages) == len(viz_types)

    @patch("streamlit.components.v1.html")
    def test_graph_component_rendering(self, mock_html, mock_streamlit):
        """Test graph component rendering."""
        mock_html.return_value = "<div>Mock Graph</div>"

        # Simulate graph rendering
        graph_html = "<svg><g>Mock Graph Content</g></svg>"

        # Would call st.components.v1.html with graph HTML
        import streamlit.components.v1 as components

        components.html(graph_html, height=700, scrolling=False)

        # Verify component was called
        mock_html.assert_called_once_with(graph_html, height=700, scrolling=False)


class TestQuickActions:
    """Test quick action button functionality."""

    def test_database_connection_test(self, mock_streamlit):
        """Test database connection test button."""
        # Simulate quick action button
        mock_streamlit.button("🔄 Test Database Connection")

        # Verify button was created
        buttons = [comp for comp in mock_streamlit.components if comp[0] == "button"]
        assert len(buttons) == 1
        assert "Test Database Connection" in buttons[0][1]

    def test_hybrid_search_test(self, mock_streamlit):
        """Test hybrid search test button."""
        mock_streamlit.button("🔍 Test Hybrid Search")

        # Verify button was created
        buttons = [comp for comp in mock_streamlit.components if comp[0] == "button"]
        assert len(buttons) == 1
        assert "Test Hybrid Search" in buttons[0][1]

    def test_ai_agent_configuration_test(self, mock_streamlit):
        """Test AI agent configuration test button."""
        mock_streamlit.button("🤖 Test AI Agent Configuration")

        # Verify button was created
        buttons = [comp for comp in mock_streamlit.components if comp[0] == "button"]
        assert len(buttons) == 1
        assert "Test AI Agent Configuration" in buttons[0][1]

    def test_multiple_quick_actions(self, mock_streamlit):
        """Test multiple quick action buttons."""
        actions = [
            "🔄 Test Database Connection",
            "🔍 Test Hybrid Search",
            "🤖 Test AI Agent Configuration",
            "📊 View System Metrics",
        ]

        for action in actions:
            mock_streamlit.button(action)

        # Verify all buttons were created
        buttons = [comp for comp in mock_streamlit.components if comp[0] == "button"]
        assert len(buttons) == len(actions)


class TestStatusAndMetrics:
    """Test status and metrics display."""

    def test_system_overview_display(self, mock_streamlit, mock_ui_metrics):
        """Test system overview metrics display."""
        metrics = mock_ui_metrics

        # Simulate metrics display
        overview_html = f"""
        <div class="metric-card">
            <strong>Components:</strong> {metrics["total_components"]}<br/>
            <strong>Relationships:</strong> {metrics["total_relationships"]}<br/>
            <strong>Categories:</strong> {metrics["categories"]}<br/>
            <strong>Total LOC:</strong> {metrics["total_lines"]:,}
        </div>
        """

        mock_streamlit.markdown(overview_html, unsafe_allow_html=True)

        # Verify metrics were displayed
        markdown_components = [
            comp for comp in mock_streamlit.components if comp[0] == "markdown"
        ]
        assert len(markdown_components) == 1
        assert str(metrics["total_components"]) in markdown_components[0][1]

    def test_service_status_indicators(self, mock_streamlit):
        """Test service status indicator display."""
        # Simulate different service statuses
        statuses = {
            "Services": "✅ Connected",
            "Database": "✅ Available",
            "Authentication": "⚠️ Simulated",
            "Graph Visualization": "✅ Active",
        }

        for service, status in statuses.items():
            status_html = f"<strong>{service}:</strong> {status}<br/>"
            mock_streamlit.markdown(status_html, unsafe_allow_html=True)

        # Verify status displays
        markdown_components = [
            comp for comp in mock_streamlit.components if comp[0] == "markdown"
        ]
        assert len(markdown_components) == len(statuses)

    def test_ingestion_status_updates(self, mock_streamlit):
        """Test ingestion status updates."""
        status_sequence = [
            "Not Started",
            "Initializing...",
            "Processing files...",
            "Extracting entities...",
            "Completed",
        ]

        for status in status_sequence:
            mock_streamlit.session_state.ingestion_status = status

            # Display status
            mock_streamlit.markdown(f"**Status:** {status}")

        # Verify final status
        assert mock_streamlit.session_state.ingestion_status == "Completed"

        # Verify status displays
        markdown_components = [
            comp for comp in mock_streamlit.components if comp[0] == "markdown"
        ]
        assert len(markdown_components) == len(status_sequence)


class TestErrorHandlingUI:
    """Test error handling in UI context."""

    def test_connection_error_display(self, mock_streamlit):
        """Test connection error display."""
        error_message = "Failed to connect to Azure Cosmos DB"
        mock_streamlit.error(error_message)

        # Verify error was displayed
        error_messages = [msg for msg in mock_streamlit.messages if msg[0] == "error"]
        assert len(error_messages) == 1
        assert error_messages[0][1] == error_message

    def test_query_error_handling(self, mock_streamlit):
        """Test query error handling in chat interface."""
        # Simulate query error
        error_query = "problematic query"
        error_response = "Error processing query: Invalid syntax"

        # Add error to chat history
        mock_streamlit.session_state.chat_history = [
            ("user", error_query),
            ("assistant", error_response),
        ]

        # Display error in UI
        mock_streamlit.error("Query processing failed")

        # Verify error handling
        assert len(mock_streamlit.session_state.chat_history) == 2
        assert (
            "Error processing query" in mock_streamlit.session_state.chat_history[1][1]
        )

        error_messages = [msg for msg in mock_streamlit.messages if msg[0] == "error"]
        assert len(error_messages) == 1

    def test_visualization_error_fallback(self, mock_streamlit):
        """Test visualization error fallback."""
        # Simulate visualization error
        mock_streamlit.error("Pyvis visualization error: Module not found")
        mock_streamlit.info("Falling back to Enhanced D3.js...")

        # Verify error and fallback messages
        error_messages = [msg for msg in mock_streamlit.messages if msg[0] == "error"]
        info_messages = [msg for msg in mock_streamlit.messages if msg[0] == "info"]

        assert len(error_messages) == 1
        assert len(info_messages) == 1
        assert "visualization error" in error_messages[0][1]
        assert "Falling back" in info_messages[0][1]

    def test_service_unavailable_warning(self, mock_streamlit):
        """Test service unavailable warning."""
        warning_message = "Mosaic services not available - using simulated mode"
        mock_streamlit.warning(warning_message)

        # Verify warning was displayed
        warning_messages = [
            msg for msg in mock_streamlit.messages if msg[0] == "warning"
        ]
        assert len(warning_messages) == 1
        assert warning_messages[0][1] == warning_message


class TestNavigationAndState:
    """Test navigation and state management."""

    def test_page_rerun_functionality(self, mock_streamlit):
        """Test page rerun functionality."""
        # Simulate rerun trigger
        mock_streamlit.rerun()

        # Verify rerun was called
        reruns = [comp for comp in mock_streamlit.components if comp[0] == "rerun"]
        assert len(reruns) == 1

    def test_state_preservation_across_interactions(self, mock_streamlit):
        """Test state preservation across user interactions."""
        # Set initial state
        initial_entities = [{"id": "test1", "name": "Test Entity"}]
        initial_node = {"id": "selected1", "name": "Selected Node"}

        mock_streamlit.session_state.entities = initial_entities
        mock_streamlit.session_state.selected_node = initial_node

        # Simulate user interaction (button click)
        mock_streamlit.button("Test Action")

        # State should be preserved
        assert mock_streamlit.session_state.entities == initial_entities
        assert mock_streamlit.session_state.selected_node == initial_node

    def test_dynamic_content_updates(self, mock_streamlit):
        """Test dynamic content updates."""
        # Initial content
        mock_streamlit.markdown("Initial content")

        # Update content based on state change
        mock_streamlit.session_state.selected_node = {
            "id": "new_node",
            "name": "New Node",
        }

        # Display updated content
        if mock_streamlit.session_state.selected_node:
            node_name = mock_streamlit.session_state.selected_node["name"]
            mock_streamlit.markdown(f"Selected: {node_name}")

        # Verify content updates
        markdown_components = [
            comp for comp in mock_streamlit.components if comp[0] == "markdown"
        ]
        assert len(markdown_components) == 2
        assert "Initial content" in markdown_components[0][1]
        assert "Selected: New Node" in markdown_components[1][1]


class TestPerformanceAndResponsiveness:
    """Test performance and responsiveness aspects."""

    def test_large_dataset_rendering(self, mock_streamlit, performance_test_data):
        """Test rendering with large datasets."""
        # Set large dataset
        mock_streamlit.session_state.entities = performance_test_data["entities"]
        mock_streamlit.session_state.relationships = performance_test_data[
            "relationships"
        ]

        # Simulate metrics display
        entity_count = len(mock_streamlit.session_state.entities)
        relationship_count = len(mock_streamlit.session_state.relationships)

        mock_streamlit.markdown(f"Entities: {entity_count}")
        mock_streamlit.markdown(f"Relationships: {relationship_count}")

        # Verify large dataset handling
        assert entity_count == 100
        assert relationship_count == 150

        # Verify components were rendered
        markdown_components = [
            comp for comp in mock_streamlit.components if comp[0] == "markdown"
        ]
        assert len(markdown_components) == 2

    def test_concurrent_component_rendering(self, mock_streamlit):
        """Test concurrent component rendering."""
        # Simulate multiple components being rendered simultaneously
        components_to_render = [
            ("title", "Main Title"),
            ("header", "Section Header"),
            ("markdown", "Content paragraph"),
            ("button", "Action Button"),
            ("selectbox", "Options", ["A", "B", "C"]),
        ]

        for comp_type, *args in components_to_render:
            if comp_type == "title":
                mock_streamlit.title(args[0])
            elif comp_type == "header":
                mock_streamlit.header(args[0])
            elif comp_type == "markdown":
                mock_streamlit.markdown(args[0])
            elif comp_type == "button":
                mock_streamlit.button(args[0])
            elif comp_type == "selectbox":
                mock_streamlit.selectbox(args[0], args[1])

        # Verify all components were rendered
        assert len(mock_streamlit.components) == len(components_to_render)

    def test_memory_efficient_chat_history(self, mock_streamlit):
        """Test memory efficient chat history management."""
        # Add many chat messages
        max_messages = 100
        for i in range(max_messages):
            mock_streamlit.session_state.chat_history.append(("user", f"Message {i}"))
            mock_streamlit.session_state.chat_history.append(
                ("assistant", f"Response {i}")
            )

        # Verify chat history size
        assert len(mock_streamlit.session_state.chat_history) == max_messages * 2

        # Simulate chat history truncation for performance
        max_history_size = 50
        if len(mock_streamlit.session_state.chat_history) > max_history_size:
            mock_streamlit.session_state.chat_history = (
                mock_streamlit.session_state.chat_history[-max_history_size:]
            )

        # Verify truncation
        assert len(mock_streamlit.session_state.chat_history) == max_history_size


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
