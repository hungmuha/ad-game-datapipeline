# Stream Downloader

A Node.js script to download video streams from CSV data using FFmpeg.

## Features

- Reads stream URLs from CSV files
- Tracks downloads in a master CSV to prevent duplicates
- Downloads .m3u8 (HLS) streams using FFmpeg
- Automatically detects and skips .mpd (DASH) files with DRM protection
- Automatic retry support for failed downloads
- Progress tracking and summary statistics

## Requirements

- Node.js (v14 or higher)
- npm

## Installation

1. Install dependencies:

```bash
npm install
```

This will install:
- `csv-parser` - CSV file parsing
- `csv-writer` - CSV file writing
- `fluent-ffmpeg` - FFmpeg wrapper
- `ffmpeg-static` - Bundled FFmpeg binary

## Input CSV Format

Your input CSV must contain at least these columns:

- `title` - Title of the stream
- `air_date` - Air date of the stream
- `stream_url` - URL to the stream (.m3u8 or .mpd)

Optional columns (will be preserved in master CSV):
- `network` - Network name
- `status` - Current status
- `extraction_time` - When the URL was extracted

Example CSV:
```csv
title,air_date,network,stream_url,status,extraction_time
Game 1,2024-01-15,ESPN,https://example.com/stream1.m3u8,pending,2024-01-15T10:00:00Z
Game 2,2024-01-16,FOX,https://example.com/stream2.mpd,pending,2024-01-16T10:00:00Z
```

## Usage

Run the script with your input CSV file:

```bash
node download-streams.js input.csv
```

Or using npm:

```bash
npm start input.csv
```

## How It Works

1. **Reads Master CSV**: Checks `master.csv` for previously downloaded/attempted streams
2. **Validates Input**: Ensures required columns exist in your input CSV
3. **Filters Duplicates**: Skips streams already in master CSV (based on title + air_date)
4. **Downloads Streams**: Uses FFmpeg to download each new stream to `./videos/`
5. **Updates Master CSV**: Records each attempt with status, absolute file path, and timestamp
6. **Retry Support**: Failed downloads can be retried by running the script again

## Duplicate Detection

The script intelligently tracks downloads by creating a unique key from `title + air_date`. This means:

- **Multiple CSVs are safe**: You can feed different CSV files without re-downloading the same games
- **Automatic deduplication**: If a game with the same title and air date exists in `master.csv`, it's automatically skipped
- **Cross-run tracking**: Progress persists across multiple script runs

Example:
```bash
# First run
node download-streams.js fubo_recordings_20260123.csv

# Later, feed another CSV - already downloaded games are automatically skipped
node download-streams.js fubo_recordings_20260124.csv
```

## Output

- **Downloaded Videos**: Saved to `./videos/` directory
  - Filename format: `{title}_{air_date}.mp4`
  - Special characters are sanitized to underscores

- **Master CSV**: `master.csv` tracks all download attempts
  - Columns: title, air_date, network, stream_url, status, file_path, filename, extraction_time, download_time
  - Status values: `downloaded`, `failed`, or `skipped_drm`
  - `file_path`: Absolute path to the downloaded MP4 file (empty if skipped or failed)
  - `filename`: Name of the downloaded MP4 file (empty if skipped or failed)

## Example

```bash
$ node download-streams.js games.csv

Input file: games.csv

Master CSV not found. Will create a new one.
Read 5 entries from input CSV.
CSV validation passed.

Found 5 new entries to process.

[1/5] Processing: Game 1 - 2024-01-15
Downloading: ./videos/Game_1_2024-01-15.mp4
Progress: 100.00%
Download completed successfully.

[2/5] Processing: Game 2 - 2024-01-16
Skipping: MPD stream detected (DRM-protected)

[3/5] Processing: Game 3 - 2024-01-17
Downloading: ./videos/Game_3_2024-01-17.mp4
Download failed: Connection timeout

...

==================================================
SUMMARY
==================================================
Total entries:           5
Skipped (existing):      0
Skipped (DRM-protected): 1
Successfully downloaded: 3
Failed downloads:        1
==================================================
```

## Retrying Failed Downloads

Failed downloads are recorded in `master.csv` with status `failed`. To retry:

1. Remove failed entries from `master.csv`, OR
2. Create a new input CSV with only the failed entries, OR
3. Modify the script to allow re-attempts of failed downloads

## DRM-Protected Content

The script automatically detects and skips .mpd (DASH) streams because they are typically DRM-protected (Widevine, PlayReady, etc.). Even if downloaded, these files would be encrypted and unplayable without decryption keys.

When a .mpd URL is detected:
- The download is skipped
- Status is set to `skipped_drm` in master.csv
- The entry is still recorded for tracking purposes

## Troubleshooting

**FFmpeg errors**: The script uses `ffmpeg-static` which includes a bundled FFmpeg binary. If you encounter issues, ensure you have a stable internet connection.

**CSV parsing errors**: Ensure your CSV file is properly formatted with headers and no empty rows.

**Permission errors**: Ensure the script has write permissions for the current directory.

## License

ISC
