# Revisit Processor

Produces revisit signals: when and where the pathologist returns to previously inspected regions.

## Inputs

- `viewport_keyframes.json` and `viewport_events.jsonl`
- optional mouse and gaze support signals

## Current Status

This module is intentionally marked as evolving/TBD. Revisit logic is heuristic and should be iterated with real sessions.

## Initial Heuristic Direction

- Define spatial cells or region clusters on slide coordinates
- Track visit intervals per region
- Mark revisit if region is visited again after:
  - a minimum time gap, and
  - at least one intervening different region
- Score revisit strength using:
  - dwell time on return
  - zoom level on return
  - supporting gaze/mouse concentration

## Outputs

1. `revisit_events.jsonl`
- revisit events with region id, first-visit timestamp, return timestamp, and strength score

2. `revisit_summary.json`
- per-case revisit statistics and ranked high-interest regions
