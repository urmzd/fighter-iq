# Fighter IQ — project tasks

# Install dependencies and shell completion
install:
    uv sync
    @just completion

# Install/reinstall shell completion for fighter-iq
completion:
    uv run fighter-iq --install-completion

# Run linter
lint:
    uv run ruff check .

# Format code
fmt:
    uv run ruff format .

# Check formatting without modifying
check-fmt:
    uv run ruff format --check .

# Quality gate: format + lint
check: check-fmt lint

# Full CI gate
ci: check-fmt lint

# Record showcase with teasr
record:
    teasr showme

# Run end-to-end sanity test: analyze a short clip, then launch review
sanity-test:
    uv run fighter-iq analyze inputs/aI3tuBrDNQY.mp4 --duration 10 --no-visualize --output outputs/sanity_test.json
    uv run fighter-iq review --analysis outputs/sanity_test.json --video inputs/aI3tuBrDNQY.mp4