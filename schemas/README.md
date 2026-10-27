# Schemas

Shared output contracts for PathLearn processors.

## Primary Contract

- `unified_event.schema.json`: common event format used by viewport/gaze/interaction/mouse modules

## Unified Event Philosophy

Most non-voice modalities should map into a tool-use style event format so multimodal traces can be consumed consistently by downstream AI training pipelines.

Voice remains a parallel aligned text stream and can be linked to unified events by timestamp and case/slide ids.
