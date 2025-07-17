# Generate Semantic Kernel Plugin Workflow

Creates the boilerplate file and test for a new Semantic Kernel plugin, and logs the corresponding task in Conport.

## Usage
`/generate-plugin <PluginName>`

## Argument
- `$ARGUMENTS`: The name of the plugin (e.g., "Ingestion").

## Chained Workflow
1.  **Plan:** Use `sequential-thinking` to outline the new plugin's purpose and key functions based on the PRD for the `$ARGUMENTS` plugin.
2.  **Create Plugin File:** Use `desktop-commander` to `write_file` at `src/mosaic/plugins/{{.Input.Arguments | toLower}}.py` with a standard Python class boilerplate for a Semantic Kernel plugin. Include `__init__` and a placeholder `@sk_function`.
3.  **Create Test File:** Use `desktop-commander` to `write_file` at `tests/test_{{.Input.Arguments | toLower}}_plugin.py` with boilerplate `pytest` setup.
4.  **Track Progress:** Use `conport` to `log_progress` with status "TODO" for the task "Implement all functions for the `$ARGUMENTS` Plugin".