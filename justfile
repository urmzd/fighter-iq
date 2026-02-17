# MMA AI Coach — project tasks

# Install dependencies and shell completion
install:
    uv sync
    @just completion

# Install/reinstall shell completion for mma-coach
completion:
    uv run mma-coach --install-completion

# Run end-to-end sanity test: analyze a short clip, then launch review
sanity-test:
    uv run mma-coach analyze inputs/aI3tuBrDNQY.mp4 --duration 10 --no-visualize --output outputs/sanity_test.json
    uv run mma-coach review --analysis outputs/sanity_test.json --video inputs/aI3tuBrDNQY.mp4