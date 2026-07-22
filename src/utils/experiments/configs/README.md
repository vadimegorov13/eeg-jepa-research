# Retained Experiment Configs

These JSON files are disabled provenance records for completed canonical experiments. The batch
runner rejects them because every entry has `"enabled": false`.

Do not re-enable an entry merely because its artifact is absent or because reproduction is
convenient. First check the workspace-root `AGENTS.md` section 2e closed-experiment registry. A new
run requires a materially new hypothesis, a prespecified protocol, and a new config filename.
