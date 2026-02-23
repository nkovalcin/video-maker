"""
Moment Detection Module
Detects interesting moments in videos based on scene changes, motion, and audio.
"""

import cv2
import numpy as np
import librosa
from typing import List, Dict, Tuple
from pathlib import Path
import json


class MomentDetector:
    """Detects interesting moments in video files."""

    def __init__(self, config: Dict):
        """
        Initialize the moment detector.

        Args:
            config: Configuration dictionary with thresholds
        """
        self.scene_threshold = config.get('scene_threshold', 30.0)
        self.motion_threshold = config.get('motion_threshold', 15.0)
        self.audio_threshold = config.get('audio_threshold', -25)
        self.min_moment_duration = config.get('min_moment_duration', 0.5)
        self.debug_mode = config.get('debug_mode', False)

    def analyze_video(self, video_path: Path) -> List[Dict]:
        """
        Analyze a video file and detect interesting moments.

        Args:
            video_path: Path to video file

        Returns:
            List of detected moments with timestamps and scores
        """
        print(f"Analyzing {video_path.name}...")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Error: Cannot open {video_path}")
            return []

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        print(f"  Duration: {duration:.2f}s, FPS: {fps:.1f}, Frames: {total_frames}")

        # For very short videos (Live Photos), analyze but use more content
        if duration < 2.5 and self.min_moment_duration < duration:
            print(f"  Short video detected - analyzing with full duration strategy")

            # Still detect moments for scoring
            visual_moments = self._detect_visual_moments(cap, fps)
            audio_moments = self._detect_audio_moments(video_path, duration)

            # Combine
            detected = self._combine_moments(visual_moments, audio_moments, str(video_path), duration)

            # If we found good moments, use them; otherwise use whole video
            if detected and max(m['score'] for m in detected) > 0.3:
                all_moments = detected
            else:
                # Use whole video with moderate score
                all_moments = [{
                    'timestamp': duration / 2,
                    'duration': duration * 0.95,
                    'score': 0.5,
                    'types': ['full_video'],
                    'intensity': 50,
                    'video_path': str(video_path),
                    'event_count': 1
                }]
        else:
            # Detect visual moments
            visual_moments = self._detect_visual_moments(cap, fps)

            # Detect audio moments
            audio_moments = self._detect_audio_moments(video_path, duration)

            # Combine and score moments
            all_moments = self._combine_moments(
                visual_moments,
                audio_moments,
                str(video_path),
                duration
            )

        cap.release()

        print(f"  Found {len(all_moments)} interesting moments")
        return all_moments

    def _detect_visual_moments(self, cap: cv2.VideoCapture, fps: float) -> List[Dict]:
        """
        Detect moments with significant visual changes.

        Args:
            cap: OpenCV video capture object
            fps: Frames per second

        Returns:
            List of visual moments with timestamps
        """
        moments = []
        prev_frame = None
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert to grayscale and resize for faster processing
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (320, 240))

            if prev_frame is not None:
                # Calculate frame difference
                diff = cv2.absdiff(prev_frame, gray)
                mean_diff = np.mean(diff)

                # Detect scene changes and motion
                if mean_diff > self.scene_threshold:
                    timestamp = frame_count / fps
                    moments.append({
                        'timestamp': timestamp,
                        'type': 'scene_change',
                        'intensity': mean_diff,
                        'frame': frame_count
                    })
                elif mean_diff > self.motion_threshold:
                    timestamp = frame_count / fps
                    moments.append({
                        'timestamp': timestamp,
                        'type': 'motion',
                        'intensity': mean_diff,
                        'frame': frame_count
                    })

            prev_frame = gray
            frame_count += 1

        # Reset video capture to beginning
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        return moments

    def _detect_audio_moments(self, video_path: Path, duration: float) -> List[Dict]:
        """
        Detect moments with significant audio activity.

        Args:
            video_path: Path to video file
            duration: Video duration in seconds

        Returns:
            List of audio moments with timestamps
        """
        try:
            # Load audio
            y, sr = librosa.load(str(video_path), sr=None, mono=True)

            # Calculate RMS energy
            hop_length = 512
            rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]

            # Convert to dB
            rms_db = librosa.amplitude_to_db(rms, ref=np.max)

            # Find peaks
            moments = []
            for i, db_value in enumerate(rms_db):
                if db_value > self.audio_threshold:
                    timestamp = (i * hop_length) / sr
                    if timestamp <= duration:
                        moments.append({
                            'timestamp': timestamp,
                            'type': 'audio_peak',
                            'intensity': db_value
                        })

            return moments

        except Exception as e:
            print(f"  Warning: Could not analyze audio: {e}")
            return []

    def _combine_moments(
        self,
        visual: List[Dict],
        audio: List[Dict],
        video_path: str,
        duration: float
    ) -> List[Dict]:
        """
        Combine visual and audio moments, merge nearby events, and calculate scores.

        Args:
            visual: List of visual moments
            audio: List of audio moments
            video_path: Path to source video
            duration: Video duration

        Returns:
            Combined and scored list of moments
        """
        # Combine all moments
        all_moments = visual + audio

        if not all_moments:
            # No moments detected, create fallback moments
            print("  No moments detected, using fallback strategy")
            return self._create_fallback_moments(video_path, duration)

        # Sort by timestamp
        all_moments.sort(key=lambda x: x['timestamp'])

        # Merge nearby moments (within 1 second)
        merged = []
        current_group = [all_moments[0]]

        for moment in all_moments[1:]:
            if moment['timestamp'] - current_group[-1]['timestamp'] < 1.0:
                current_group.append(moment)
            else:
                # Score and add the group
                merged.append(self._score_moment_group(current_group, video_path))
                current_group = [moment]

        # Add last group
        if current_group:
            merged.append(self._score_moment_group(current_group, video_path))

        # Sort by score
        merged.sort(key=lambda x: x['score'], reverse=True)

        return merged

    def _score_moment_group(self, group: List[Dict], video_path: str) -> Dict:
        """
        Calculate a combined score for a group of nearby moments.

        Args:
            group: List of moments to combine
            video_path: Source video path

        Returns:
            Dictionary with combined moment info and score
        """
        avg_timestamp = np.mean([m['timestamp'] for m in group])

        # Count different types of events
        types = [m['type'] for m in group]
        has_scene_change = 'scene_change' in types
        has_motion = 'motion' in types
        has_audio = 'audio_peak' in types

        # Calculate intensity
        intensities = [m.get('intensity', 0) for m in group]
        avg_intensity = np.mean(intensities)
        max_intensity = np.max(intensities)

        # Score calculation (0-1 scale)
        score = 0.0

        # Type bonuses
        if has_scene_change:
            score += 0.4
        if has_motion:
            score += 0.2
        if has_audio:
            score += 0.3

        # Intensity bonus (normalized)
        intensity_bonus = min(avg_intensity / 100.0, 0.3)
        score += intensity_bonus

        # Multiple event bonus
        if len(set(types)) > 1:
            score += 0.2

        # Normalize to 0-1
        score = min(score, 1.0)

        return {
            'timestamp': avg_timestamp,
            'duration': 1.0,  # Default duration
            'score': score,
            'types': list(set(types)),
            'intensity': avg_intensity,
            'video_path': video_path,
            'event_count': len(group)
        }

    def _create_fallback_moments(self, video_path: str, duration: float) -> List[Dict]:
        """
        Create fallback moments when no interesting moments are detected.

        Args:
            video_path: Source video path
            duration: Video duration

        Returns:
            List of evenly distributed moments
        """
        moments = []
        num_moments = min(5, int(duration / 2))

        for i in range(num_moments):
            timestamp = (i + 1) * (duration / (num_moments + 1))
            moments.append({
                'timestamp': timestamp,
                'duration': 1.0,
                'score': 0.5,
                'types': ['fallback'],
                'intensity': 0,
                'video_path': video_path,
                'event_count': 0
            })

        return moments
