"""
Video Editor Module
Handles video composition, slow-motion effects, and final rendering.
"""

try:
    from moviepy.editor import VideoFileClip, concatenate_videoclips, vfx
except ImportError:
    from moviepy import VideoFileClip, concatenate_videoclips, vfx
from typing import List, Dict, Tuple
from pathlib import Path
import numpy as np


class VideoEditor:
    """Composes final video from selected moments with slow-motion effects."""

    def __init__(self, config: Dict):
        """
        Initialize the video editor.

        Args:
            config: Configuration dictionary
        """
        self.target_duration = config.get('target_duration', 10)
        self.transition_duration = config.get('transition_duration', 0.3)
        self.slowmo_range = config.get('slowmo_factor_range', [2.0, 3.0])
        self.output_fps = config.get('output_fps', 30)
        self.output_resolution = tuple(config.get('output_resolution', [1920, 1080]))
        self.max_moments = config.get('max_moments', 8)

    def select_moments(self, all_moments: List[Dict], config: Dict) -> List[Dict]:
        """
        Select the best moments that fit into target duration.
        Ensures no static/boring moments and strong hook in first 3 seconds.

        Args:
            all_moments: All detected moments sorted by score
            config: Configuration dictionary

        Returns:
            Selected moments with assigned slow-motion factors
        """
        if not all_moments:
            print("Warning: No moments to select from")
            return []

        print(f"\nSelecting moments for {self.target_duration}s video...")

        # Filter out low-score moments
        min_score = config.get('min_moment_score', 0.15)
        filtered_moments = [m for m in all_moments if m['score'] >= min_score]

        if config.get('prefer_high_motion', False):
            # Prioritize moments with motion
            motion_moments = [m for m in filtered_moments if 'motion' in m['types'] or 'scene_change' in m['types']]
            if motion_moments:
                filtered_moments = motion_moments + [m for m in filtered_moments if m not in motion_moments]

        print(f"  Filtered to {len(filtered_moments)} moments (score >= {min_score})")

        # HOOK STRATEGY: Find best moment for first 3 seconds
        hook_moment = None
        if config.get('require_hook_in_first_3s', False):
            # Find highest scoring moment with motion
            motion_moments = [m for m in filtered_moments if 'motion' in m['types'] or 'scene_change' in m['types']]
            if motion_moments:
                hook_moment = motion_moments[0]
                print(f"  🎣 HOOK selected: Score {hook_moment['score']:.2f} | Types: {', '.join(hook_moment['types'])}")
                filtered_moments = [m for m in filtered_moments if m != hook_moment]

        selected = []
        remaining_time = self.target_duration

        # Add hook first (aim for 2-3s hook)
        if hook_moment:
            ideal_hook = config.get('ideal_clip_duration', 2.0)
            hook_duration = min(hook_moment.get('duration', 1.0), ideal_hook, self.target_duration * 0.3)
            hook_moment['slowmo_factor'] = 1.0
            hook_moment['actual_duration'] = hook_duration
            hook_moment['duration'] = hook_duration
            selected.append(hook_moment)
            remaining_time -= hook_duration
            print(f"  ✓ Hook: Score {hook_moment['score']:.2f} | {hook_duration:.1f}s | Types: {', '.join(hook_moment['types'])}")

        # Fill remaining time with best moments
        for moment in filtered_moments:
            if remaining_time <= 0.3:  # Need at least 0.3s for a clip
                break

            # Determine slow-motion factor
            if moment['score'] > 0.7:
                slowmo_factor = self.slowmo_range[1]
                moment_type = "slow-motion"
            elif moment['score'] > 0.5:
                slowmo_factor = np.mean(self.slowmo_range)
                moment_type = "medium slow-motion"
            else:
                slowmo_factor = 1.0
                moment_type = "normal speed"

            # Calculate duration - aim for ideal clip length
            max_duration = config.get('max_moment_duration', 3.0)
            ideal = config.get('ideal_clip_duration', 2.0)

            # Target 2s per clip for good pacing
            base_duration = min(moment.get('duration', 1.0), ideal, max_duration, remaining_time)
            actual_duration = base_duration * slowmo_factor

            # Check if we have time
            if actual_duration <= remaining_time and actual_duration >= 0.3:
                moment['slowmo_factor'] = slowmo_factor
                moment['actual_duration'] = actual_duration
                moment['duration'] = base_duration
                selected.append(moment)
                remaining_time -= actual_duration

                print(f"  ✓ Score: {moment['score']:.2f} | {moment_type} ({slowmo_factor:.1f}x) | "
                      f"{actual_duration:.1f}s | Types: {', '.join(moment['types'])}")
            elif remaining_time >= 0.5:
                # Use remaining time
                moment['actual_duration'] = remaining_time
                moment['slowmo_factor'] = 1.0
                moment['duration'] = remaining_time
                selected.append(moment)
                print(f"  ✓ Score: {moment['score']:.2f} | final clip {remaining_time:.1f}s")
                break

        total_duration = sum(m['actual_duration'] for m in selected)
        print(f"\n✓ Selected {len(selected)} moments, total: {total_duration:.1f}s / {self.target_duration}s")

        if total_duration < self.target_duration * 0.8:
            print(f"  ⚠️  Warning: Only {total_duration:.1f}s of content (target: {self.target_duration}s)")

        return selected

    def create_video(
        self,
        selected_moments: List[Dict],
        output_path: Path,
        timeline_path: Path
    ) -> bool:
        """
        Create the final video from selected moments.

        Args:
            selected_moments: List of moments to include
            output_path: Where to save the output video
            timeline_path: Where to save the timeline JSON

        Returns:
            True if successful, False otherwise
        """
        if not selected_moments:
            print("Error: No moments selected for video creation")
            return False

        print(f"\nCreating final video...")

        try:
            clips = []
            timeline = []

            for i, moment in enumerate(selected_moments):
                print(f"  Processing moment {i+1}/{len(selected_moments)}...")

                # Load video clip
                video = VideoFileClip(moment['video_path'])

                # Extract the moment
                start_time = max(0, moment['timestamp'] - moment['duration'] / 2)
                end_time = min(video.duration, start_time + moment['duration'])
                # Use subclipped for moviepy 2.x, fallback to subclip for 1.x
                if hasattr(video, 'subclipped'):
                    clip = video.subclipped(start_time, end_time)
                else:
                    clip = video.subclip(start_time, end_time)

                # Apply slow-motion if needed
                slowmo_factor = moment.get('slowmo_factor', 1.0)
                if slowmo_factor > 1.0:
                    # Slow down video by changing FPS
                    print(f"    Applying {slowmo_factor:.1f}x slow-motion")
                    try:
                        # MoviePy 2.x approach: slow down by reducing playback speed
                        original_fps = clip.fps
                        new_fps = original_fps / slowmo_factor
                        clip = clip.with_fps(new_fps)
                        # Adjust duration
                        clip = clip.with_updated_frame_t(lambda t: t / slowmo_factor)
                    except Exception as e:
                        print(f"    Warning: Could not apply slow-motion: {e}")

                # Ensure clip doesn't exceed allocated duration
                if clip.duration > moment['actual_duration']:
                    if hasattr(clip, 'subclipped'):
                        clip = clip.subclipped(0, moment['actual_duration'])
                    else:
                        clip = clip.subclip(0, moment['actual_duration'])

                # Add crossfade transition (except for first clip)
                # Note: Crossfade temporarily disabled for MoviePy 2.x compatibility
                if i > 0 and self.transition_duration > 0:
                    try:
                        if hasattr(clip, 'crossfadein'):
                            # MoviePy 1.x
                            clip = clip.crossfadein(self.transition_duration)
                        else:
                            # MoviePy 2.x - fade in at start
                            if hasattr(clip, 'with_effects'):
                                from moviepy.video.fx import FadeIn
                                clip = clip.with_effects([FadeIn(self.transition_duration)])
                    except Exception as e:
                        print(f"    Note: Crossfade not applied ({e})")

                # Verify clip is valid before adding
                if clip is not None and hasattr(clip, 'duration') and clip.duration > 0:
                    clips.append(clip)
                    print(f"    Added clip: {clip.duration:.2f}s")
                else:
                    print(f"    Warning: Skipping invalid clip")

                # Save to timeline
                timeline.append({
                    'index': i,
                    'source_video': Path(moment['video_path']).name,
                    'source_timestamp': moment['timestamp'],
                    'duration': moment['actual_duration'],
                    'slowmo_factor': slowmo_factor,
                    'score': moment['score'],
                    'types': moment['types']
                })

                video.close()

            # Extract clips using FFmpeg directly
            print(f"  Extracting {len(clips)} clips using FFmpeg...")
            temp_files = []
            import subprocess

            for i, moment in enumerate(selected_moments):
                temp_file = output_path.parent / f"temp_clip_{i}.mp4"

                # Calculate timing - use actual_duration for precise control
                clip_duration = moment.get('actual_duration', moment.get('duration', 1.0))
                start_time = max(0, moment['timestamp'] - clip_duration / 2)

                # Use FFmpeg to extract clip with precise duration
                cmd = [
                    'ffmpeg', '-y',
                    '-ss', str(start_time),  # Seek before input for faster processing
                    '-i', moment['video_path'],
                    '-t', str(clip_duration),  # Exact duration
                    '-c:v', 'libx264',
                    '-c:a', 'aac',
                    '-preset', 'fast',
                    '-avoid_negative_ts', 'make_zero',
                    str(temp_file)
                ]

                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    temp_files.append(temp_file)
                    # Verify extracted duration
                    verify_cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                                '-of', 'default=noprint_wrappers=1:nokey=1', str(temp_file)]
                    verify_result = subprocess.run(verify_cmd, capture_output=True, text=True)
                    actual_dur = float(verify_result.stdout.strip()) if verify_result.returncode == 0 else 0
                    print(f"    Extracted clip {i+1}/{len(selected_moments)}: requested {clip_duration:.2f}s, got {actual_dur:.2f}s")
                else:
                    print(f"    Warning: Failed to extract clip {i+1}: {result.stderr[:100]}")

            # Close all moviepy clips
            for clip in clips:
                try:
                    clip.close()
                except:
                    pass

            # Create FFmpeg concat file
            concat_file = output_path.parent / "concat_list.txt"
            with open(concat_file, 'w') as f:
                for temp_file in temp_files:
                    # Use absolute path for safety
                    f.write(f"file '{temp_file.absolute()}'\n")

            print(f"  Concat list:")
            with open(concat_file, 'r') as f:
                print(f"    {f.read()}")

            # Concatenate with FFmpeg
            print(f"  Combining {len(clips)} clips with FFmpeg...")
            import subprocess
            result = subprocess.run([
                'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
                '-i', str(concat_file),
                '-c', 'copy',
                str(output_path)
            ], capture_output=True, text=True)

            if result.returncode != 0:
                print(f"    FFmpeg concat failed, trying re-encode...")
                result = subprocess.run([
                    'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
                    '-i', str(concat_file),
                    '-c:v', 'libx264', '-c:a', 'aac',
                    str(output_path)
                ], capture_output=True, text=True)

            # Cleanup temp files
            for temp_file in temp_files:
                temp_file.unlink()
            concat_file.unlink()

            # Load final clip to get duration info
            final_clip = VideoFileClip(str(output_path))
            final_duration = final_clip.duration

            if final_duration < self.target_duration:
                print(f"  Note: Final duration {final_duration:.1f}s is shorter than target {self.target_duration}s")

            # Save timeline
            import json
            # Convert numpy types to native Python
            timeline_serializable = []
            for m in timeline:
                moment_dict = {}
                for key, value in m.items():
                    if hasattr(value, 'item'):  # numpy types
                        moment_dict[key] = value.item()
                    elif isinstance(value, (list, tuple)):
                        moment_dict[key] = [v.item() if hasattr(v, 'item') else v for v in value]
                    else:
                        moment_dict[key] = value
                timeline_serializable.append(moment_dict)

            with open(timeline_path, 'w') as f:
                json.dump({
                    'target_duration': self.target_duration,
                    'actual_duration': float(final_clip.duration),
                    'num_clips': len(temp_files),
                    'moments': timeline_serializable
                }, f, indent=2)

            # Cleanup
            final_clip.close()
            for clip in clips:
                clip.close()

            print(f"\n✅ Video created successfully!")
            print(f"   Output: {output_path}")
            print(f"   Duration: {final_clip.duration:.2f}s")
            print(f"   Timeline: {timeline_path}")

            return True

        except Exception as e:
            print(f"\n❌ Error creating video: {e}")
            import traceback
            traceback.print_exc()
            return False

    def create_debug_visualization(
        self,
        moments: List[Dict],
        output_dir: Path
    ):
        """
        Create debug visualization showing detected moments.

        Args:
            moments: All detected moments
            output_dir: Directory to save debug files
        """
        try:
            import cv2

            print("\nCreating debug visualizations...")

            for i, moment in enumerate(moments[:10]):  # Limit to top 10
                video = cv2.VideoCapture(moment['video_path'])
                video.set(cv2.CAP_PROP_POS_MSEC, moment['timestamp'] * 1000)

                ret, frame = video.read()
                if ret:
                    # Add text overlay
                    text = f"Score: {moment['score']:.2f} | {', '.join(moment['types'])}"
                    cv2.putText(
                        frame,
                        text,
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )

                    # Save frame
                    output_file = output_dir / f"moment_{i+1:02d}_score_{moment['score']:.2f}.jpg"
                    cv2.imwrite(str(output_file), frame)

                video.release()

            print(f"  Saved debug frames to {output_dir}")

        except Exception as e:
            print(f"  Warning: Could not create debug visualization: {e}")
