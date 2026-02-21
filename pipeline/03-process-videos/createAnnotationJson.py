import re
import json
import argparse
import os
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional


def timecode_to_seconds(timecode: str, fps: int = 24) -> float:
    """
    Convert timecode HH:MM:SS:FF to seconds.
    
    Args:
        timecode: Timecode string in format HH:MM:SS:FF
        fps: Frames per second (default 24, adjust based on your video)
    
    Returns:
        Time in seconds as float
    """
    parts = timecode.split(':')
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = int(parts[2])
    frames = int(parts[3])
    
    total_seconds = hours * 3600 + minutes * 60 + seconds + (frames / fps)
    return round(total_seconds, 2)


def parse_edl(file_path: str, fps: int = 24) -> List[Dict]:
    """
    Parse EDL file and extract markers with timecodes and colors.
    
    Args:
        file_path: Path to the EDL file
        fps: Frames per second for timecode conversion
    
    Returns:
        List of marker dictionaries
    """
    markers = []
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Look for event lines (e.g., "001  001      V     C        00:01:16:07...")
        if line and line[0].isdigit():
            # Parse the timecode line
            parts = line.split()
            if len(parts) >= 8:
                # Extract start timecode (5th element)
                start_timecode = parts[4]
                
                # Check next line for marker metadata
                if i + 1 < len(lines):
                    metadata_line = lines[i + 1].strip()
                    
                    # Extract color and marker name
                    color_match = re.search(r'\|C:(ResolveColor\w+)', metadata_line)
                    marker_match = re.search(r'\|M:([^|]+)', metadata_line)
                    
                    if color_match and marker_match:
                        color = color_match.group(1)
                        
                        # Determine label based on color
                        label = "ad" if color == "ResolveColorRed" else "content"
                        
                        markers.append({
                            'timecode': start_timecode,
                            'time_seconds': timecode_to_seconds(start_timecode, fps),
                            'color': color,
                            'label': label
                        })
        
        i += 1
    
    return markers


def load_master_map(master_map_path: str = 'master_map.csv') -> Optional[pd.DataFrame]:
    """
    Load the master map CSV file.
    
    Args:
        master_map_path: Path to master_map.csv file
    
    Returns:
        DataFrame with master map data, or None if file doesn't exist
    """
    master_map_file = Path(master_map_path)
    if not master_map_file.exists():
        print(f"Warning: Master map file not found at {master_map_path}")
        return None
    
    try:
        df = pd.read_csv(master_map_file)
        return df
    except Exception as e:
        print(f"Error loading master map: {e}")
        return None


def find_video_in_master_map(edl_path: str, master_map_df: pd.DataFrame) -> Optional[Dict]:
    """
    Find video metadata in master map by EDL path.
    
    Args:
        edl_path: Path to the EDL file
        master_map_df: DataFrame containing master map data
    
    Returns:
        Dictionary with video metadata, or None if not found
    """
    if master_map_df is None:
        return None
    
    # Normalize the EDL path for comparison
    edl_path_normalized = str(Path(edl_path))
    
    # Try to find by exact EDL path match
    for _, row in master_map_df.iterrows():
        if str(Path(row['edl_path'])) == edl_path_normalized:
            return {
                'video_name': row['video_name'],
                'video_path': row['video_path'],
                'edl_path': row['edl_path'],
                'duration_seconds': float(row['duration_seconds'])
            }
    
    # Try to find by video name extracted from EDL path
    edl_stem = Path(edl_path).stem
    # Remove common suffixes like '_marker'
    video_name_guess = edl_stem.replace('_marker', '')
    
    for _, row in master_map_df.iterrows():
        if row['video_name'] == video_name_guess:
            return {
                'video_name': row['video_name'],
                'video_path': row['video_path'],
                'edl_path': row['edl_path'],
                'duration_seconds': float(row['duration_seconds'])
            }
    
    return None


def create_segments(markers: List[Dict], video_duration: float = None) -> Tuple[List[Dict], List[float]]:
    """
    Create segments from markers.
    
    Args:
        markers: List of markers sorted by time
        video_duration: Total video duration in seconds (optional)
    
    Returns:
        Tuple of (segments list, change_points list)
    """
    if not markers:
        return [], []
    
    # Sort markers by time
    markers_sorted = sorted(markers, key=lambda x: x['time_seconds'])
    
    segments = []
    change_points = []
    
    # Create segments between markers
    for i in range(len(markers_sorted)):
        start_time = markers_sorted[i]['time_seconds']
        end_time = markers_sorted[i + 1]['time_seconds'] if i + 1 < len(markers_sorted) else video_duration
        
        label = markers_sorted[i]['label']
        segment_type = "commercial_break" if label == "ad" else "game_play"
        
        segment = {
            'segment_id': i,
            'start_time': start_time,
            'end_time': end_time,
            'label': label,
            'type': segment_type
        }
        
        segments.append(segment)
        change_points.append(start_time)
    
    # Add the final end time as a change point if exists
    if segments and segments[-1]['end_time']:
        change_points.append(segments[-1]['end_time'])
    
    return segments, change_points


def create_silent_segments(markers: List[Dict]) -> List[Dict]:
    """
    Create silent_segments array from markers.
    
    Args:
        markers: List of markers with timecode, time_seconds, color, and label
    
    Returns:
        List of silent segment dictionaries
    """
    if not markers:
        return []
    
    # Sort markers by time
    markers_sorted = sorted(markers, key=lambda x: x['time_seconds'])
    
    # Filter out markers with timestamp 0.0
    filtered_markers = [m for m in markers_sorted if m['time_seconds'] != 0.0]
    
    silent_segments = []
    
    # Create silent segment for each marker
    for i, marker in enumerate(filtered_markers):
        # Determine is_CP based on color: Yellow = 0, all others = 1
        is_cp = 0 if marker['color'] == 'ResolveColorYellow' else 1
        
        silent_segment = {
            'segment_id': f"silence_{i+1:03d}",
            'timestamp': marker['time_seconds'],
            'is_CP': is_cp,
            'notes': ""
        }
        
        silent_segments.append(silent_segment)
    
    return silent_segments


def createAnnotationJson(
    file_path: str, 
    video_id: str = None, 
    video_duration: float = None, 
    fps: int = 24, 
    output_path: str = None,
    use_master_map: bool = False,
    master_map_path: str = 'master_map.csv'
):
    """
    Create annotation JSON file from EDL file.
    
    Args:
        file_path: Path to the EDL file
        video_id: Video identifier (defaults to filename without extension)
        video_duration: Total video duration in seconds (optional)
        fps: Frames per second for timecode conversion
        output_path: Path to save JSON file (optional, defaults to video manifest folder)
        use_master_map: Whether to use master_map.csv for metadata
        master_map_path: Path to master_map.csv file
    """
    # If using master map, try to load video metadata
    if use_master_map:
        master_map_df = load_master_map(master_map_path)
        if master_map_df is not None:
            video_metadata = find_video_in_master_map(file_path, master_map_df)
            if video_metadata:
                video_id = video_id or video_metadata['video_name']
                video_duration = video_duration or video_metadata['duration_seconds']
                print(f"Loaded metadata from master map for: {video_id}")
                print(f"  Duration: {video_duration:.2f} seconds")
            else:
                print(f"Warning: Could not find video metadata in master map for {file_path}")
    
    # Parse EDL file
    markers = parse_edl(file_path, fps)
    
    # Create silent segments
    silent_segments = create_silent_segments(markers)
    
    # Create segments
    segments, change_points = create_segments(markers, video_duration)
    
    # Generate video_id from filename if not provided
    if not video_id:
        video_id = Path(file_path).stem
        # Remove common suffixes like '_marker'
        video_id = video_id.replace('_marker', '')
    
    # Determine output path if not specified
    if not output_path:
        # Save in the video's manifest folder
        edl_path = Path(file_path)
        
        # Check if EDL is in a manifest subfolder structure
        if edl_path.parent.name != 'manifests' and edl_path.parent.parent.name == 'manifests':
            # EDL is in manifests/video_name/ structure
            manifest_folder = edl_path.parent
        else:
            # EDL is directly in manifests or elsewhere
            # Create a subfolder for this video
            manifest_folder = edl_path.parent / video_id
            manifest_folder.mkdir(parents=True, exist_ok=True)
        
        output_path = manifest_folder / f"{video_id}_annotation.json"
    
    # Create annotation structure
    annotation = {
        'video_id': video_id,
        'duration_seconds': video_duration,
        'silent_segments': silent_segments,
        'segments': segments,
        'change_points': change_points
    }
    
    # Save to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(annotation, f, indent=2)
    print(f"Annotation saved to: {output_path}")
    
    return annotation


#  important: need to pass in video id and video duration (or use --use-master-map)
def main():
    parser = argparse.ArgumentParser(
        description='Convert EDL file to annotation JSON',
        epilog='Example: python createAnnotationJson.py --file_path manifests/video/video_marker.edl --use-master-map'
    )
    parser.add_argument('--file_path', type=str, required=True, help='Path to EDL file')
    parser.add_argument('--video_id', type=str, help='Video identifier (optional if using --use-master-map)')
    parser.add_argument('--duration', type=float, help='Video duration in seconds (optional if using --use-master-map)')
    parser.add_argument('--fps', type=int, default=24, help='Frames per second (default: 24)')
    parser.add_argument('--output', type=str, help='Output JSON file path (default: saves in video manifest folder)')
    parser.add_argument('--use-master-map', action='store_true', 
                        help='Load video metadata from master_map.csv')
    parser.add_argument('--master-map-path', type=str, default='master_map.csv',
                        help='Path to master_map.csv file (default: master_map.csv)')
    
    args = parser.parse_args()
    
    createAnnotationJson(
        file_path=args.file_path,
        video_id=args.video_id,
        video_duration=args.duration,
        fps=args.fps,
        output_path=args.output,
        use_master_map=args.use_master_map,
        master_map_path=args.master_map_path
    )


if __name__ == "__main__":
    main()