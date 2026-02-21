#!/usr/bin/env python3
"""
Utility script to add start_time_hms and end_time_hms columns to existing manifests.
Run this once to update all existing CSV files.
"""

import pandas as pd
from pathlib import Path


def seconds_to_hms(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def update_manifest(manifest_path: Path):
    """Add HMS columns to an existing manifest."""
    print(f"Updating {manifest_path.name}...")
    
    # Read existing CSV
    df = pd.read_csv(manifest_path)
    
    # Check if HMS columns already exist
    if 'start_time_hms' in df.columns and 'end_time_hms' in df.columns:
        print(f"  ✓ Already has HMS columns, skipping.")
        return
    
    # Add HMS columns
    df['start_time_hms'] = df['silent_start_time'].apply(seconds_to_hms)
    df['end_time_hms'] = df['silent_end_time'].apply(seconds_to_hms)
    
    # Reorder columns to put HMS after silent times
    columns = [
        'pair_id',
        'video_file_path',
        'audio_file_path',
        'silent_start_time',
        'silent_end_time',
        'start_time_hms',
        'end_time_hms',
        'CP'
    ]
    
    df = df[columns]
    
    # Save back to CSV
    df.to_csv(manifest_path, index=False)
    print(f"  ✓ Updated with {len(df)} rows.")


def main():
    """Update all manifests in the manifests/ folder."""
    manifests_dir = Path('manifests')
    
    if not manifests_dir.exists():
        print("No manifests folder found.")
        return
    
    manifest_files = list(manifests_dir.glob('*.csv'))
    
    if not manifest_files:
        print("No CSV files found in manifests/ folder.")
        return
    
    print(f"Found {len(manifest_files)} manifest file(s) to update.\n")
    
    for manifest_path in manifest_files:
        update_manifest(manifest_path)
    
    print(f"\n✅ Done! Updated {len(manifest_files)} manifest(s).")


if __name__ == '__main__':
    main()



