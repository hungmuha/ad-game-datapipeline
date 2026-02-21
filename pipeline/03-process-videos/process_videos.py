#!/usr/bin/env python3
"""
CP Detection Data Pipeline
Processes NFL game videos to extract audio, detect silence segments, and generate
CSV manifests for Change Point (CP) model training.
"""

import argparse
import logging
import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
from pydub import AudioSegment
from scipy import signal
from scipy.io import wavfile
import cv2


def setup_logging():
    """Configure logging system with timestamps."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def get_video_fps(video_path: str) -> float:
    """
    Get the frame rate (FPS) of a video file.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Frame rate as float, defaults to 59.94 if unable to detect
    """
    try:
        video = cv2.VideoCapture(video_path)
        fps = video.get(cv2.CAP_PROP_FPS)
        video.release()
        
        if fps > 0:
            logging.info(f"Detected video FPS: {fps}")
            return fps
        else:
            logging.warning(f"Could not detect FPS, defaulting to 59.94")
            return 59.94
            
    except Exception as e:
        logging.error(f"Failed to detect FPS from {video_path}: {e}, defaulting to 59.94")
        return 59.94


def extract_audio(video_path: str, output_audio_path: str, sample_rate: int = 16000) -> Optional[float]:
    """
    Extract audio track from video file and save as WAV.
    
    Args:
        video_path: Path to input video file
        output_audio_path: Path to output WAV file
        sample_rate: Target sample rate in Hz (default: 16000)
        
    Returns:
        Duration of audio in seconds, or None if extraction failed
    """
    try:
        logging.info(f"Extracting audio from {Path(video_path).name}")
        audio = AudioSegment.from_file(video_path)
        
        # Convert to mono and set sample rate
        audio = audio.set_channels(1)
        audio = audio.set_frame_rate(sample_rate)
        
        # Export as WAV
        audio.export(output_audio_path, format='wav')
        
        duration = len(audio) / 1000.0  # Convert ms to seconds
        logging.info(f"Audio extracted: {duration:.2f}s duration")
        return duration
        
    except Exception as e:
        logging.error(f"Failed to extract audio from {video_path}: {e}")
        return None


def detect_silence_segments(
    audio_path: str,
    filter_low: int = 300,
    filter_high: int = 6000,
    min_silence_ms: int = 10,
    volume_threshold: int = 4
) -> List[Tuple[float, float]]:
    """
    Detect silence segments in audio file with band-pass filtering.
    
    Args:
        audio_path: Path to WAV audio file
        filter_low: Low frequency cutoff for band-pass filter (Hz)
        filter_high: High frequency cutoff for band-pass filter (Hz)
        min_silence_ms: Minimum silence duration in milliseconds
        volume_threshold: Volume threshold after normalization to int16
        
    Returns:
        List of (start_time, end_time) tuples in seconds
    """
    try:
        # Load WAV file
        sample_rate, audio_data = wavfile.read(audio_path)
        
        # Convert to float for processing
        if audio_data.dtype == np.int16:
            audio_float = audio_data.astype(np.float32)
        else:
            audio_float = audio_data.astype(np.float32)
        
        # Design Butterworth band-pass filter
        nyquist = sample_rate / 2
        low = filter_low / nyquist
        high = filter_high / nyquist
        
        # Ensure filter frequencies are valid
        low = max(0.01, min(low, 0.99))
        high = max(low + 0.01, min(high, 0.99))
        
        sos = signal.butter(5, [low, high], btype='band', output='sos')
        filtered_audio = signal.sosfilt(sos, audio_float)
        
        # Normalize to int16 range
        max_val = np.max(np.abs(filtered_audio))
        if max_val > 0:
            normalized = (filtered_audio / max_val) * 32767
        else:
            normalized = filtered_audio
        
        # Detect silence: where absolute amplitude < threshold
        is_silent = np.abs(normalized) < volume_threshold
        
        # Find contiguous silent segments
        silence_segments = []
        in_silence = False
        silence_start = 0
        
        for i, silent in enumerate(is_silent):
            if silent and not in_silence:
                # Start of silence
                in_silence = True
                silence_start = i
            elif not silent and in_silence:
                # End of silence
                in_silence = False
                silence_end = i
                
                # Convert samples to time
                start_time = silence_start / sample_rate
                end_time = silence_end / sample_rate
                duration_ms = (end_time - start_time) * 1000
                
                # Filter by minimum duration
                if duration_ms >= min_silence_ms:
                    silence_segments.append((start_time, end_time))
        
        # Handle case where audio ends in silence
        if in_silence:
            end_time = len(is_silent) / sample_rate
            start_time = silence_start / sample_rate
            duration_ms = (end_time - start_time) * 1000
            if duration_ms >= min_silence_ms:
                silence_segments.append((start_time, end_time))
        
        logging.info(f"Detected {len(silence_segments)} silence segments")
        return silence_segments
        
    except Exception as e:
        logging.error(f"Failed to detect silence in {audio_path}: {e}")
        return []


def extend_segment_times(
    start: float,
    end: float,
    extension: float = 2.2,
    video_duration: Optional[float] = None
) -> Tuple[float, float]:
    """
    Extend segment times by adding buffer before and after, with boundary clamping.
    
    Args:
        start: Original start time in seconds
        end: Original end time in seconds
        extension: Time to extend before/after in seconds (default: 2.2)
        video_duration: Total video duration for boundary clamping (optional)
        
    Returns:
        Tuple of (extended_start, extended_end) in seconds
    """
    extended_start = start - extension
    extended_end = end + extension
    
    # Clamp to boundaries if video duration provided
    if video_duration is not None:
        extended_start = max(0.0, extended_start)
        extended_end = min(video_duration, extended_end)
    
    return extended_start, extended_end


def seconds_to_hms(seconds: float) -> str:
    """
    Convert seconds to HH:MM:SS.mmm format with milliseconds.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Time string in HH:MM:SS.mmm format
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def seconds_to_edl_timecode(seconds: float, fps: float = 59.94) -> str:
    """
    Convert seconds to EDL timecode format HH:MM:SS:FF.
    
    Args:
        seconds: Time in seconds
        fps: Frames per second (default: 59.94)
        
    Returns:
        Timecode string in HH:MM:SS:FF format
    """
    # Calculate total frames first to avoid rounding issues
    total_frames = round(seconds * fps)
    
    # Extract components from total frames
    frames_per_hour = int(fps * 3600)
    frames_per_minute = int(fps * 60)
    
    hours = total_frames // frames_per_hour
    remaining_frames = total_frames % frames_per_hour
    
    minutes = remaining_frames // frames_per_minute
    remaining_frames = remaining_frames % frames_per_minute
    
    secs = remaining_frames // int(fps)
    frames = remaining_frames % int(fps)
    
    return f"{hours:02d}:{minutes:02d}:{secs:02d}:{frames:02d}"


def add_frames(seconds: float, frames: int = 1, fps: float = 59.94) -> float:
    """
    Add frames to a time value in seconds.
    
    Args:
        seconds: Time in seconds
        frames: Number of frames to add (default: 1)
        fps: Frames per second (default: 59.94)
        
    Returns:
        New time in seconds
    """
    return seconds + (frames / fps)


def create_manifest(
    video_file: str,
    audio_file: str,
    original_segments: List[Tuple[float, float]],
    extended_segments: List[Tuple[float, float]],
    output_manifest_path: str
) -> str:
    """
    Generate CSV manifest with silence segment metadata.
    
    Args:
        video_file: Path to video file
        audio_file: Path to audio file
        original_segments: List of original (start_time, end_time) tuples before extension
        extended_segments: List of extended (start_time, end_time) tuples after extension
        output_manifest_path: Path to output CSV file
        
    Returns:
        Path to created manifest file
    """
    # Generate sequential IDs and build records
    records = []
    for idx, (orig_seg, ext_seg) in enumerate(zip(original_segments, extended_segments), start=1):
        orig_start, orig_end = orig_seg
        ext_start, ext_end = ext_seg
        
        records.append({
            'id': idx,
            'video_file_path': str(Path(video_file).absolute()),
            'audio_file_path': str(Path(audio_file).absolute()),
            'silent_start_time': orig_start,
            'silent_end_time': orig_end,
            'silent_start_time_hms': seconds_to_hms(orig_start),
            'silent_end_time_hms': seconds_to_hms(orig_end),
            'start_extended': ext_start,
            'end_extended': ext_end,
            'start_extended_hms': seconds_to_hms(ext_start),
            'end_extended_hms': seconds_to_hms(ext_end),
            'CP': ''
        })
    
    # Create DataFrame with specified columns
    df = pd.DataFrame(records, columns=[
        'id',
        'video_file_path',
        'audio_file_path',
        'silent_start_time',
        'silent_end_time',
        'silent_start_time_hms',
        'silent_end_time_hms',
        'start_extended',
        'end_extended',
        'start_extended_hms',
        'end_extended_hms',
        'CP'
    ])
    
    # Save to CSV
    df.to_csv(output_manifest_path, index=False)
    logging.info(f"Manifest created: {output_manifest_path} ({len(records)} segments)")
    
    return output_manifest_path


def create_edl_file(
    video_file: str,
    original_segments: List[Tuple[float, float]],
    output_edl_path: str,
    fps: float = 59.94
) -> str:
    """
    Generate EDL (Edit Decision List) file for video editing software.
    
    Args:
        video_file: Path to video file
        original_segments: List of original (start_time, end_time) tuples
        output_edl_path: Path to output EDL file
        fps: Frames per second for timecode conversion (default: 59.94)
        
    Returns:
        Path to created EDL file
    """
    video_stem = Path(video_file).stem
    
    # EDL header
    edl_lines = [
        f"TITLE: {video_stem}",
        "FCM: NON-DROP FRAME",
        ""
    ]
    
    # First entry: marker at 0.0 seconds
    source_in_0 = seconds_to_edl_timecode(0.0, fps)
    source_out_0 = seconds_to_edl_timecode(add_frames(0.0, frames=1, fps=fps), fps)
    entry_0 = f"001  001      V     C        {source_in_0} {source_out_0} {source_in_0} {source_out_0}  "
    edl_lines.append(entry_0)
    metadata_0 = f" |C:ResolveColorBlue |M:Marker 1 |D:1"
    edl_lines.append(metadata_0)
    edl_lines.append("")
    
    # Generate EDL entries for detected silence segments
    for idx, (start_time, end_time) in enumerate(original_segments, start=2):
        # Convert start time to timecode
        source_in = seconds_to_edl_timecode(start_time, fps)
        record_in = source_in
        
        # Add 1 frame for OUT timecodes
        start_plus_1frame = add_frames(start_time, frames=1, fps=fps)
        source_out = seconds_to_edl_timecode(start_plus_1frame, fps)
        record_out = source_out
        
        # Format: ID  REEL  TRACK  EDIT_TYPE  SOURCE_IN SOURCE_OUT RECORD_IN RECORD_OUT
        entry = f"{idx:03d}  001      V     C        {source_in} {source_out} {record_in} {record_out}  "
        edl_lines.append(entry)
        
        # Add metadata line (color marker without name)
        metadata = f" |C:ResolveColorBlue |M:Marker {idx} |D:1"
        edl_lines.append(metadata)
        edl_lines.append("")
    
    # Write EDL file
    with open(output_edl_path, 'w') as f:
        f.write('\n'.join(edl_lines))
    
    logging.info(f"EDL file created: {output_edl_path} ({len(original_segments) + 1} markers)")
    
    return output_edl_path


def update_master_map(
    video_name: str,
    video_path: str,
    edl_path: str,
    duration_seconds: float,
    master_map_path: str = 'master_map.csv'
) -> None:
    """
    Update or create master_map.csv with video metadata.
    
    Args:
        video_name: Name of the video (without extension)
        video_path: Absolute path to the video file
        edl_path: Relative path to the EDL file from project root
        duration_seconds: Duration of the video in seconds
        master_map_path: Path to master map CSV file (default: 'master_map.csv')
    """
    try:
        # Check if master map exists
        master_map_file = Path(master_map_path)
        
        if master_map_file.exists():
            # Read existing master map
            df = pd.read_csv(master_map_file)
            
            # Check if video already exists in the map
            if video_name in df['video_name'].values:
                # Update existing entry
                df.loc[df['video_name'] == video_name, 'video_path'] = video_path
                df.loc[df['video_name'] == video_name, 'edl_path'] = edl_path
                df.loc[df['video_name'] == video_name, 'duration_seconds'] = duration_seconds
                logging.info(f"Updated master_map.csv entry for {video_name}")
            else:
                # Append new entry
                new_row = pd.DataFrame([{
                    'video_name': video_name,
                    'video_path': video_path,
                    'edl_path': edl_path,
                    'duration_seconds': duration_seconds
                }])
                df = pd.concat([df, new_row], ignore_index=True)
                logging.info(f"Added {video_name} to master_map.csv")
        else:
            # Create new master map
            df = pd.DataFrame([{
                'video_name': video_name,
                'video_path': video_path,
                'edl_path': edl_path,
                'duration_seconds': duration_seconds
            }])
            logging.info(f"Created master_map.csv with entry for {video_name}")
        
        # Save master map
        df.to_csv(master_map_file, index=False)
        
    except Exception as e:
        logging.error(f"Failed to update master_map.csv: {e}")


def process_video_folder(
    input_folder: str,
    audio_output: str = 'audio',
    manifest_output: str = 'manifests',
    config: Optional[Dict] = None
) -> Dict[str, int]:
    """
    Process all video files in a folder.
    
    Args:
        input_folder: Path to folder containing .mp4 files
        audio_output: Output folder for audio files (default: 'audio')
        manifest_output: Output folder for manifest files (default: 'manifests')
        config: Configuration dictionary with processing parameters
        
    Returns:
        Dictionary with processing statistics
    """
    if config is None:
        config = {}
    
    # Extract configuration parameters with defaults
    filter_low = config.get('filter_low', 300)
    filter_high = config.get('filter_high', 6000)
    min_silence_ms = config.get('min_silence_ms', 10)
    volume_threshold = config.get('volume_threshold', 4)
    time_extension = config.get('time_extension', 2.2)
    sample_rate = config.get('sample_rate', 16000)
    
    # Create output directories
    audio_dir = Path(audio_output)
    manifest_dir = Path(manifest_output)
    
    try:
        audio_dir.mkdir(parents=True, exist_ok=True)
        manifest_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logging.error(f"Failed to create output directories: {e}")
        return {'processed': 0, 'skipped': 0, 'failed': 0}
    
    # Scan for .mp4 files
    input_path = Path(input_folder)
    video_files = list(input_path.glob('*.mp4'))
    
    if not video_files:
        logging.warning(f"No .mp4 files found in {input_folder}")
        return {'processed': 0, 'skipped': 0, 'failed': 0}
    
    logging.info(f"Found {len(video_files)} video file(s) to process")
    
    # Process statistics
    stats = {'processed': 0, 'skipped': 0, 'failed': 0}
    
    for video_file in video_files:
        video_stem = video_file.stem

        # Create output folder for this video
        video_output_dir = manifest_dir / video_stem
        video_output_dir.mkdir(parents=True, exist_ok=True)

        manifest_path = video_output_dir / f"{video_stem}.csv"
        edl_path = video_output_dir / f"{video_stem}_marker.edl"
        audio_file = audio_dir / f"{video_stem}.wav"

        # Skip if already fully processed (all 3 files exist)
        if manifest_path.exists() and edl_path.exists() and audio_file.exists():
            logging.warning(f"Skipping {video_file.name} (already processed - all outputs exist)")
            stats['skipped'] += 1
            continue

        logging.info(f"Processing {video_file.name}")

        # Detect video FPS
        video_fps = get_video_fps(str(video_file))

        # Extract or reuse audio
        if audio_file.exists():
            logging.info(f"Reusing existing audio file: {audio_file.name}")
            try:
                # Get duration from existing audio file
                sample_rate_temp, audio_data = wavfile.read(str(audio_file))
                duration = len(audio_data) / sample_rate_temp
                logging.info(f"Audio duration: {duration:.2f}s")
            except Exception as e:
                logging.error(f"Failed to read existing audio file {audio_file}: {e}")
                logging.info(f"Re-extracting audio from video")
                duration = extract_audio(str(video_file), str(audio_file), sample_rate)
                if duration is None:
                    stats['failed'] += 1
                    continue
        else:
            # Extract audio from video
            duration = extract_audio(str(video_file), str(audio_file), sample_rate)
            if duration is None:
                stats['failed'] += 1
                continue
        
        # Detect silence segments
        silence_segments = detect_silence_segments(
            str(audio_file),
            filter_low=filter_low,
            filter_high=filter_high,
            min_silence_ms=min_silence_ms,
            volume_threshold=volume_threshold
        )
        
        if not silence_segments:
            logging.warning(f"No silence segments detected in {video_file.name}")
        
        # Extend time windows and clamp to boundaries
        extended_segments = []
        for start, end in silence_segments:
            ext_start, ext_end = extend_segment_times(
                start, end,
                extension=time_extension,
                video_duration=duration
            )
            extended_segments.append((ext_start, ext_end))
        
        # Create manifest with both original and extended segments
        try:
            create_manifest(
                str(video_file),
                str(audio_file),
                silence_segments,  # Original segments
                extended_segments,  # Extended segments
                str(manifest_path)
            )
            
            # Create EDL file with original segments
            create_edl_file(
                str(video_file),
                silence_segments,  # Original segments only
                str(edl_path),
                fps=video_fps
            )
            
            # Update master map with video metadata
            # Use relative path for EDL from project root
            try:
                # Try to get relative path from current working directory
                edl_relative_path = str(edl_path.resolve().relative_to(Path.cwd().resolve()))
            except ValueError:
                # If EDL is outside cwd, try relative to project root
                project_root = Path.cwd().resolve()
                # Walk up to find project root (look for common markers)
                while project_root.parent != project_root:
                    if (project_root / 'pipeline').exists() or (project_root / 'data').exists():
                        try:
                            edl_relative_path = str(edl_path.resolve().relative_to(project_root))
                            break
                        except ValueError:
                            pass
                    project_root = project_root.parent
                else:
                    # Fall back to absolute path if relative path cannot be determined
                    edl_relative_path = str(edl_path.resolve())
                    logging.warning(f"Could not determine relative path for EDL, using absolute: {edl_relative_path}")
            
            update_master_map(
                video_name=video_stem,
                video_path=str(video_file.absolute()),
                edl_path=edl_relative_path,
                duration_seconds=duration
            )
            
            stats['processed'] += 1
        except Exception as e:
            logging.error(f"Failed to create manifest/EDL for {video_file.name}: {e}")
            stats['failed'] += 1
    
    # Log summary
    logging.info(f"Processing complete: {stats['processed']} processed, "
                f"{stats['skipped']} skipped, {stats['failed']} failed")
    
    return stats


def main():
    """Main entry point with CLI argument parsing."""
    setup_logging()
    
    parser = argparse.ArgumentParser(
        description='CP Detection Data Pipeline: Extract audio and detect silence segments from videos'
    )
    
    parser.add_argument(
        '--input',
        required=True,
        help='Input folder path containing .mp4 files'
    )
    parser.add_argument(
        '--audio-output',
        default='audio',
        help='Audio output folder (default: audio)'
    )
    parser.add_argument(
        '--manifest-output',
        default='manifests',
        help='Manifest output folder (default: manifests)'
    )
    parser.add_argument(
        '--filter-low',
        type=int,
        default=300,
        help='Band-pass filter low frequency in Hz (default: 300)'
    )
    parser.add_argument(
        '--filter-high',
        type=int,
        default=6000,
        help='Band-pass filter high frequency in Hz (default: 6000)'
    )
    parser.add_argument(
        '--min-silence-ms',
        type=int,
        default=10,
        help='Minimum silence duration in milliseconds (default: 10)'
    )
    parser.add_argument(
        '--volume-threshold',
        type=int,
        default=4,
        help='Volume threshold after normalization to int16 (default: 4)'
    )
    parser.add_argument(
        '--time-extension',
        type=float,
        default=2.2,
        help='Seconds to extend before/after silence (default: 2.2)'
    )
    parser.add_argument(
        '--sample-rate',
        type=int,
        default=16000,
        help='Audio sample rate in Hz (default: 16000)'
    )
    
    args = parser.parse_args()
    
    # Build configuration dictionary
    config = {
        'filter_low': args.filter_low,
        'filter_high': args.filter_high,
        'min_silence_ms': args.min_silence_ms,
        'volume_threshold': args.volume_threshold,
        'time_extension': args.time_extension,
        'sample_rate': args.sample_rate
    }
    
    logging.info("Starting CP Detection Data Pipeline")
    logging.info(f"Input folder: {args.input}")
    logging.info(f"Configuration: {config}")
    
    # Process videos
    stats = process_video_folder(
        args.input,
        audio_output=args.audio_output,
        manifest_output=args.manifest_output,
        config=config
    )
    
    logging.info("Pipeline execution complete")


if __name__ == '__main__':
    main()

