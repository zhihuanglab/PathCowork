# Interaction Processor

Normalizes user and AI-assist interaction logs into a structured tool-use stream.

## Inputs

- Viewer interaction logs (`log.jsonl` or equivalent)
- Tool metadata (tool name/version/parameters), if available

## Processing Goals

- Parse raw interaction events
- Standardize event types across tools
- Preserve arguments, outputs, and status
- Convert interaction history into agent-training-friendly tool-use records

## Outputs

1. `interaction_events.jsonl`
- unified tool-use style events for user/AI tool invocation

2. `interaction_summary.json`
- optional rollup: counts, durations, failures, common tool chains

## Example Tool-Use Concepts

- detection model invocation
- segmentation tool usage
- annotation edit operations
