# Documentation Compliance Check Workflow

Verify that all implemented Functional Requirements (FRs) in the code are reflected in the PRD and TDD documentation, and vice-versa.

## Chained Workflow
1.  **Scan Code for FRs:** Use `desktop-commander` to `grep` for all unique occurrences of "FR-" in the `src/` directory.
2.  **Scan Docs for FRs:** Use `desktop-commander` to `grep` for all unique occurrences of "FR-" in the `docs/` directory.
3.  **Analyze & Report Discrepancies:** Use `sequential-thinking` to compare the two lists. Report on any requirements that are implemented in the code but not fully documented, or requirements that are documented but not yet implemented. This helps ensure our documentation never goes stale.