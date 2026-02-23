# Video Maker

AI-powered automatic video editor that creates engaging 10-second highlight reels from raw footage. Uses computer vision and audio analysis to find the most interesting moments.

## Features

- **Scene change detection** — identifies when something new happens
- **Motion detection** — finds moments with significant movement
- **Audio peak detection** — detects sounds, voices, action
- **Auto scoring** — ranks moments by "interestingness" (0–1 scale)
- **Hook strategy** — places the best moment in the first 3 seconds
- **No dead moments** — filters out static/boring scenes
- **Optimal pacing** — 2-second clips for variety and engagement

## How It Works

1. **Detect** — scans video for scene changes, motion, and audio peaks
2. **Score** — ranks each moment by engagement potential
3. **Select** — picks the best moments that fit the target duration
4. **Edit** — assembles the final video with transitions

## Usage

```bash
# Place input videos in the input/ folder
python3 main.py

# Output will be in the output/ folder
```

## Configuration

Edit `config.json` to customize:

| Setting | Default | Description |
|---------|---------|-------------|
| `target_duration` | `10` | Target output length in seconds |
| `scene_threshold` | `20.0` | Sensitivity for scene detection |
| `motion_threshold` | `8.0` | Sensitivity for motion detection |
| `audio_threshold` | `-35` | Audio peak detection (dB) |
| `min_moment_duration` | `0.3` | Minimum clip length |
| `max_moment_duration` | `3.0` | Maximum clip length |
| `require_hook_in_first_3s` | `true` | Force best moment as hook |
| `debug_mode` | `true` | Save frame previews |

## Project Structure

```
├── main.py        # Entry point
├── detector.py    # Moment detection (scene, motion, audio)
├── editor.py      # Video assembly and rendering
├── config.json    # Configuration
├── input/         # Drop raw videos here
└── output/        # Rendered highlights
```

## Requirements

- Python 3.8+
- FFmpeg
- OpenCV (`pip install opencv-python`)
- MoviePy (`pip install moviepy`)

## License

MIT
