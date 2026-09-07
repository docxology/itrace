Live-capture subpackage: `server.py` (dashboard HTTP server), `analysis.py`
(live-window analysis), `state.py` (session state). The deliberately thin,
hardware-fragile webcam/MediaPipe shell — keep isolated from the pure kernel.
Session artifacts land in `../../../output/live-html/`.
