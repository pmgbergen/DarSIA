# AGENTS

Tests are tiered into: fast, integration, slow.

Default agent runs must exclude slow tests and run fast + integration only.

Use these commands exactly:
- Fast + integration (default): pytest -m "fast or integration"
- Fast only: pytest -m "fast"
- Integration only: pytest -m "integration"
- Full suite (CI/PR required): pytest -m "fast or integration or slow"

CI/PR validation must run the full suite, including slow tests.
