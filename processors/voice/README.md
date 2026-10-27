# Voice Processor

Transcribes pathologist narration and aligns text with navigation context.

## Inputs

- Audio recording (for example `audio.webm`, `audio.mp3`)
- Optional viewport timeline for alignment

## Processing Goals

- Run privacy-preserving offline ASR when required
- Support optional online ASR backend
- Segment transcripts into timestamped utterances/sentences
- Align transcript segments with viewport context/time windows

## Outputs

1. `transcript.json`
- timestamped segments with text and confidence

2. `transcript_aligned.json`
- transcript linked to viewport keyframes/events

## Current Implementation

No new voice algorithm is introduced. The processor wraps existing ASR backends and normalizes output.

- Class API: `processors/voice/VoiceData.py`
- CLI entrypoint: `processors/voice/transcribe_audio.py`

### Supported backends

1. `openai_api`
- Uses OpenAI speech-to-text API
- Model example: `gpt-4o-mini-transcribe`

2. `local_whisper`
- Uses local `faster-whisper`
- Model example: `large-v3`
- Keeps audio/transcription processing on local machine for privacy-sensitive workflows

### Unified output format

Both backends produce the same JSON keys:
- `output_format`
- `backend`
- `source_audio`
- `created_at`
- `task`
- `language`
- `duration`
- `text`
- `segments`

Each segment includes:
- `id`
- `text`
- `start`
- `end`
- `avg_logprob`
- `language`
- `speaker`
- `words`

## Example Audio

- `processors/voice/example_data/audio.webm`

## CLI Examples

Local whisper:

```bash
python3 processors/voice/transcribe_audio.py \
  --input processors/voice/example_data/audio.webm \
  --output processors/voice/example_data/example_transcript.local.json \
  --backend local_whisper \
  --model large-v3
```

OpenAI API:

```bash
python3 processors/voice/transcribe_audio.py \
  --input processors/voice/example_data/audio.webm \
  --output processors/voice/example_data/example_transcript.openai.json \
  --backend openai_api \
  --model gpt-4o-mini-transcribe
```

## Offline ASR Option (Whisper)

Use an offline Whisper backend (for example Whisper large-v3-turbo) so data can stay local.

Minimal workflow:
1. Download model weights locally.
2. Run local inference on `audio.webm`.
3. Export timestamped transcript JSON.

Keep this module configurable so teams can choose offline-only or online APIs.

## Note

Voice output is not required to be converted to tool-use events; it is preserved as aligned reasoning text.
