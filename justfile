# Fight Analyzer — project tasks

# Install dependencies and shell completion
install:
    uv sync
    @just completion

# Install/reinstall shell completion for fight-analyzer
completion:
    uv run fight-analyzer --install-completion

# Run end-to-end sanity test: analyze a short clip, then launch review
sanity-test:
    uv run fight-analyzer analyze inputs/aI3tuBrDNQY.mp4 --duration 10 --no-visualize --output outputs/sanity_test.json
    uv run fight-analyzer review --analysis outputs/sanity_test.json --video inputs/aI3tuBrDNQY.mp4