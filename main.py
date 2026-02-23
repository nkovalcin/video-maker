#!/usr/bin/env python3
"""
Automatic Video Editor
Creates a 10-second highlight video from multiple input videos.
"""

import json
import sys
from pathlib import Path
from typing import List, Dict

from detector import MomentDetector
from editor import VideoEditor


def load_config(config_path: Path = Path("config.json")) -> Dict:
    """
    Load configuration from JSON file.

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary
    """
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)

    with open(config_path, 'r') as f:
        return json.load(f)


def find_input_videos(input_dir: Path = Path("input")) -> List[Path]:
    """
    Find all video files in the input directory.

    Args:
        input_dir: Directory containing input videos

    Returns:
        List of video file paths
    """
    if not input_dir.exists():
        print(f"Error: Input directory not found at {input_dir}")
        sys.exit(1)

    video_extensions = {'.mov', '.MOV', '.mp4', '.MP4', '.avi', '.AVI'}
    videos = [
        f for f in input_dir.iterdir()
        if f.is_file() and f.suffix in video_extensions
    ]

    return sorted(videos)


def main():
    """Main execution function."""
    print("=" * 60)
    print("🎬 AUTOMATIC VIDEO EDITOR")
    print("=" * 60)

    # Load configuration
    print("\n📋 Loading configuration...")
    config = load_config()
    print(f"  Target duration: {config['target_duration']}s")
    print(f"  Slow-motion range: {config['slowmo_factor_range'][0]:.1f}x - "
          f"{config['slowmo_factor_range'][1]:.1f}x")
    print(f"  Output resolution: {config['output_resolution'][0]}x{config['output_resolution'][1]}")
    print(f"  Output FPS: {config['output_fps']}")

    # Find input videos
    print("\n🔍 Searching for input videos...")
    input_videos = find_input_videos()

    if not input_videos:
        print("❌ No video files found in input/ directory!")
        print("   Please add .MOV or .MP4 files to the input/ folder.")
        sys.exit(1)

    print(f"  Found {len(input_videos)} video(s):")
    for video in input_videos:
        print(f"    - {video.name}")

    # Initialize detector
    print("\n🔎 Initializing moment detector...")
    detector = MomentDetector(config)

    # Analyze all videos
    print("\n📊 Analyzing videos for interesting moments...")
    all_moments = []
    for video in input_videos:
        moments = detector.analyze_video(video)
        all_moments.extend(moments)

    if not all_moments:
        print("\n⚠️  No interesting moments detected!")
        print("   The script will create a video using fallback moments.")

    print(f"\n📈 Total moments detected: {len(all_moments)}")

    # Save all detected moments
    moments_output = Path("output/moments_detected.json")
    with open(moments_output, 'w') as f:
        # Convert Path objects and numpy types to native Python types for JSON serialization
        moments_serializable = []
        for m in all_moments:
            moment_copy = {}
            for key, value in m.items():
                if key == 'video_path':
                    moment_copy[key] = str(value)
                elif hasattr(value, 'item'):  # numpy types
                    moment_copy[key] = value.item()
                elif isinstance(value, list):
                    moment_copy[key] = [str(v) if hasattr(v, '__fspath__') else v for v in value]
                else:
                    moment_copy[key] = value
            moments_serializable.append(moment_copy)

        json.dump({
            'total_moments': len(all_moments),
            'moments': moments_serializable
        }, f, indent=2)
    print(f"  Saved moment analysis to {moments_output}")

    # Initialize editor
    print("\n✂️  Initializing video editor...")
    editor = VideoEditor(config)

    # Select best moments
    selected_moments = editor.select_moments(all_moments, config)

    if not selected_moments:
        print("❌ Could not select any moments for video creation!")
        sys.exit(1)

    # Create debug visualization if enabled
    if config.get('debug_mode', False):
        debug_dir = Path("output/debug_frames")
        debug_dir.mkdir(parents=True, exist_ok=True)
        editor.create_debug_visualization(all_moments, debug_dir)

    # Create final video
    output_path = Path("output/final_video.mp4")
    timeline_path = Path("output/timeline.json")

    success = editor.create_video(selected_moments, output_path, timeline_path)

    if success:
        print("\n" + "=" * 60)
        print("✅ SUCCESS!")
        print("=" * 60)
        print(f"\n📹 Your video is ready: {output_path}")
        print(f"📋 Timeline saved to: {timeline_path}")
        print(f"📊 Moment analysis: {moments_output}")

        if config.get('debug_mode', False):
            print(f"🐛 Debug frames: output/debug_frames/")

        print("\n💡 Tips:")
        print("  - Adjust thresholds in config.json for different results")
        print("  - Check timeline.json to see which moments were selected")
        print("  - Review debug frames to understand moment detection")
    else:
        print("\n❌ Video creation failed. Check the error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
