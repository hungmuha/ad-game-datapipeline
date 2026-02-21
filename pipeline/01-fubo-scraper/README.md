# Fubo NFL Recordings Scraper

A Python web scraping tool that extracts video stream links (.m3u8 URLs) from Fubo TV NFL recordings. The scraper logs into Fubo TV, navigates to your NFL recordings, captures the network traffic for each game to extract the video stream URL, and exports all data to a CSV file.

## Features

- **Automated Login**: Securely logs into Fubo TV using credentials from `.env` file
- **Network Traffic Interception**: Uses selenium-wire to capture .m3u8 stream URLs from network requests
- **Batch Processing**: Processes all NFL recordings with progress tracking
- **Smart Error Handling**: Continues processing even if individual games fail
- **CSV Export**: Exports game title, air date, network, and stream URL to CSV
- **Partial Results Saving**: Saves progress if interrupted
- **FFmpeg Integration**: Test stream downloads with built-in ffmpeg support
- **Flexible CLI**: Multiple command-line options for testing and customization

## Prerequisites

- Python 3.8 or higher
- Google Chrome browser (latest version)
- Active Fubo TV subscription
- ffmpeg (optional, for testing downloads)

## Installation

### 1. Clone or Download the Repository

```bash
cd /path/to/your/projects
git clone <repository-url>
cd fubo-scraper
```

### 2. Create Virtual Environment (Recommended)

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- `selenium-wire` - For intercepting network requests
- `selenium` - Web automation framework
- `webdriver-manager` - Automatic ChromeDriver management
- `pandas` - Data manipulation and CSV export
- `python-dotenv` - Environment variable management
- `tqdm` - Progress bars
- `coloredlogs` - Enhanced logging

### 4. Install FFmpeg (Optional)

For testing stream downloads:

**macOS:**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install ffmpeg
```

**Windows:**
Download from [ffmpeg.org](https://ffmpeg.org/download.html)

## Configuration

### 1. Create Environment File

Copy the example environment file:

```bash
cp .env.example .env
```

### 2. Edit .env File

Open `.env` and add your Fubo TV credentials:

```env
# Fubo TV Credentials
FUBO_USERNAME=your_email@example.com
FUBO_PASSWORD=your_password

# Fubo TV NFL Recordings URL
FUBO_RECORDINGS_URL=https://www.fubo.tv/p/my-stuff/recordings/leagueId/191277

# Browser Settings
HEADLESS_MODE=True

# Scraper Settings
DELAY_BETWEEN_GAMES=2
IMPLICIT_WAIT=10
EXPLICIT_WAIT=15
```

**Configuration Options:**

- `FUBO_USERNAME`: Your Fubo TV email address
- `FUBO_PASSWORD`: Your Fubo TV password
- `FUBO_RECORDINGS_URL`: Direct URL to NFL recordings page
- `HEADLESS_MODE`: Run browser without GUI (True/False)
- `DELAY_BETWEEN_GAMES`: Seconds to wait between processing games
- `IMPLICIT_WAIT`: Selenium implicit wait timeout
- `EXPLICIT_WAIT`: Selenium explicit wait timeout

## Usage

### Basic Usage

Run the scraper with default settings:

```bash
python main.py
```

### Command-Line Options

```bash
# Run in headless mode (no browser window)
python main.py --headless

# Test with first 5 games only
python main.py --limit 5

# Test ffmpeg download of first stream (30 seconds)
python main.py --test-download

# Enable verbose debug logging
python main.py --verbose

# Custom output filename
python main.py --output my_recordings.csv

# Use a custom recordings URL (overrides .env config)
python main.py --url "https://www.fubo.tv/p/my-stuff/recordings/leagueId/191277"

# Combine options
python main.py --headless --limit 10 --verbose
```

### Full CLI Help

```bash
python main.py --help
```

## Output

### CSV File Format

The scraper generates a CSV file in the `output/` directory with the following columns:

| Column | Description | Example |
|--------|-------------|---------|
| title | Game title | "San Francisco 49ers at Indianapolis Colts" |
| air_date | When the game aired | "Dec 22" |
| network | Broadcasting network | "ESPN" |
| stream_url | .m3u8 stream URL | "https://..." |
| status | Extraction status | "SUCCESS" / "FAILED" / "ERROR" |
| extraction_time | When data was extracted | "2024-12-23 14:30:00" |

**Filename Format:** `fubo_recordings_YYYYMMDD_HHMMSS.csv`

### Log File

Detailed logs are saved to `output/scraper.log` including:
- Login attempts
- Navigation steps
- Each game processed
- Errors and warnings
- Network requests captured

## How It Works

1. **Initialization**: Sets up selenium-wire Chrome driver with network interception
2. **Login**: Navigates to Fubo TV and logs in with provided credentials
3. **Navigation**: Goes directly to NFL recordings page
4. **Game Discovery**: Scrolls page to load all recordings, extracts metadata
5. **Stream Capture**: For each game:
   - Clicks the game card to start playback
   - Waits for video player to initialize
   - Monitors network traffic for .m3u8 URLs
   - Extracts the master playlist URL
   - Closes the player and moves to next game
6. **Export**: Saves all data to timestamped CSV file

## Technical Details

### Why Selenium-Wire?

Regular Selenium cannot intercept network traffic. The .m3u8 stream URLs are **not** in the HTML DOM - they only appear in network requests when the video player loads. Selenium-wire extends Selenium to capture these requests.

### Stream URL Patterns

The scraper looks for URLs containing:
- `finite.m3u8`
- `media.m3u8`
- `playlist.m3u8`
- `master.m3u8`
- `.mpd` files

And filters for Fubo's CDN domains.

### CSS Selectors

The scraper uses multiple fallback selectors to find game cards:
- `div[data-testid*='recording']`
- `div[class*='recording-card']`
- `a[href*='/watch/']`
- And several others as fallbacks

If the page structure changes, you may need to inspect the DOM and update selectors in `scraper.py`.

## Downloading Streams

Once you have the .m3u8 URLs, you can download the full games using ffmpeg:

```bash
ffmpeg -i "https://stream-url.m3u8" -c copy output.mp4
```

**Example with quality options:**

```bash
# Download with re-encoding (slower, but more compatible)
ffmpeg -i "https://stream-url.m3u8" -c:v libx264 -c:a aac output.mp4

# Download specific duration (e.g., 10 minutes)
ffmpeg -i "https://stream-url.m3u8" -t 600 -c copy output.mp4

# Download with headers if needed
ffmpeg -headers "User-Agent: Mozilla/5.0" -i "https://stream-url.m3u8" -c copy output.mp4
```

## Troubleshooting

### Login Issues

**Problem:** Login fails or hangs

**Solutions:**
- Verify credentials in `.env` file
- Try running without `--headless` to see what's happening
- Check if Fubo TV requires 2FA (currently not supported)
- Ensure you have an active subscription

### No Games Found

**Problem:** Script says "No games found"

**Solutions:**
- Verify the `FUBO_RECORDINGS_URL` is correct
- Check that you actually have NFL recordings on your account
- Run without `--headless` to see the page
- Update CSS selectors if Fubo changed their page structure

### Stream URLs Not Captured

**Problem:** Games process but stream URLs show "NOT FOUND"

**Solutions:**
- Increase `DELAY_BETWEEN_GAMES` in `.env` (try 3-5 seconds)
- Check network connection (streams need to start loading)
- Run with `--verbose` to see network requests
- Ensure selenium-wire is installed correctly

### ChromeDriver Issues

**Problem:** ChromeDriver version mismatch or not found

**Solutions:**
- The script uses `webdriver-manager` which auto-downloads correct version
- Ensure Chrome browser is updated to latest version
- Try uninstalling and reinstalling selenium and webdriver-manager

### Anti-Bot Detection

**Problem:** Fubo TV blocks or challenges the scraper

**Solutions:**
- Increase delays between actions
- Run in non-headless mode (less detectable)
- The scraper already uses realistic user-agent and timings
- Add random delays: modify `DELAY_BETWEEN_GAMES`

### Partial Results

**Problem:** Script crashes mid-run

**Solution:**
- Partial results are automatically saved as `fubo_recordings_partial_*.csv`
- Check `output/scraper.log` for error details
- Use `--limit` to test with fewer games first

## Best Practices

1. **Test First**: Always run with `--limit 3` before processing all games
2. **Headless Mode**: Use `--headless` for faster processing (after testing)
3. **Monitor Logs**: Check `output/scraper.log` for issues
4. **Respect Rate Limits**: Don't decrease delays too much
5. **Backup Credentials**: Keep `.env` secure and backed up
6. **Check CSV**: Verify a few URLs work before processing all games

## Project Structure

```
fubo-scraper/
├── .env                    # Your credentials (git-ignored)
├── .env.example            # Template for credentials
├── .gitignore              # Excluded files
├── requirements.txt        # Python dependencies
├── config.py               # Configuration management
├── scraper.py              # Main FuboScraper class
├── main.py                 # CLI entry point
├── output/                 # CSV and log files
│   ├── .gitkeep
│   ├── fubo_recordings_*.csv
│   └── scraper.log
└── README.md               # This file
```

## FAQ

### Q: How long does it take to process all games?

**A:** With 137 games and 2-second delay between each, approximately **4-5 minutes**. The actual time depends on network speed and page load times.

### Q: Are the stream URLs permanent?

**A:** Stream URLs may expire after some time. It's recommended to download games soon after extracting URLs.

### Q: Can I run this on a schedule?

**A:** Yes, you can set up a cron job (Linux/Mac) or Task Scheduler (Windows) to run the script automatically.

### Q: Will this work with other sports/channels?

**A:** The scraper is specifically designed for NFL recordings. For other content, you'd need to modify the `FUBO_RECORDINGS_URL` and potentially adjust the game extraction logic.

### Q: Is this legal?

**A:** This tool is for personal use to access content from your own paid Fubo TV subscription. Respect Fubo's Terms of Service and copyright laws.

### Q: Can I contribute?

**A:** Yes! Feel free to submit issues or pull requests for improvements.

## Error Codes

The CSV `status` column indicates:
- **SUCCESS**: Stream URL successfully captured
- **FAILED**: Clicked game but no stream URL found
- **ERROR**: Exception occurred while processing game

## Performance Tips

1. **Headless Mode**: 10-15% faster than GUI mode
2. **Limit Concurrent Processing**: Don't run multiple instances
3. **Network**: Fast, stable internet connection recommended
4. **Resources**: Close other Chrome instances while running

## Security Notes

- Never commit `.env` file to version control
- `.env` is already in `.gitignore`
- Use environment variables for sensitive data
- Keep your Fubo credentials secure

## Changelog

### v1.0.0 (2024-12-23)
- Initial release
- Selenium-wire integration for network interception
- Automated login and navigation
- Batch processing with progress bars
- CSV export with metadata
- Partial results saving
- FFmpeg test download support
- Comprehensive error handling

## License

This project is for educational and personal use only. Respect Fubo TV's Terms of Service and applicable copyright laws.

## Support

For issues, questions, or contributions:
1. Check the Troubleshooting section above
2. Review `output/scraper.log` for detailed errors
3. Open an issue with logs and error messages

---

**Happy Scraping!**
