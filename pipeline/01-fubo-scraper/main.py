#!/usr/bin/env python3
"""
Fubo NFL Scraper - CLI Entry Point
Extract video stream URLs from Fubo TV NFL recordings
"""
import argparse
import sys
import os
import subprocess
from datetime import datetime

from scraper import FuboScraper
from config import Config


def test_ffmpeg_download(stream_url: str, output_file: str = "test_download.mp4", duration: int = 30):
    """
    Test downloading a stream with ffmpeg

    Args:
        stream_url: The .m3u8 stream URL
        output_file: Output filename
        duration: Duration to download in seconds
    """
    print(f"\n{'='*60}")
    print("Testing ffmpeg download...")
    print(f"Stream URL: {stream_url[:80]}...")
    print(f"Duration: {duration} seconds")
    print(f"Output: {output_file}")
    print(f"{'='*60}\n")

    # Check if ffmpeg is installed
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("ERROR: ffmpeg is not installed or not in PATH")
        print("Install ffmpeg: brew install ffmpeg (macOS) or apt-get install ffmpeg (Linux)")
        return

    # Build ffmpeg command
    cmd = [
        'ffmpeg',
        '-i', stream_url,
        '-t', str(duration),
        '-c', 'copy',
        '-y',  # Overwrite output file
        output_file
    ]

    try:
        print("Running ffmpeg...")
        result = subprocess.run(cmd, capture_output=False, text=True)

        if result.returncode == 0:
            print(f"\n✓ Successfully downloaded {duration}s to {output_file}")
            # Get file size
            size_mb = os.path.getsize(output_file) / (1024 * 1024)
            print(f"File size: {size_mb:.2f} MB")
        else:
            print(f"\n✗ Download failed with return code {result.returncode}")

    except Exception as e:
        print(f"✗ Error during download: {e}")


def print_banner():
    """Print application banner"""
    banner = """
    ╔════════════════════════════════════════════════════════╗
    ║                                                        ║
    ║           FUBO NFL RECORDINGS SCRAPER                  ║
    ║           Extract Stream URLs from Fubo TV             ║
    ║                                                        ║
    ╚════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_summary(results: list):
    """Print scraping summary"""
    print(f"\n{'='*60}")
    print("SCRAPING SUMMARY")
    print(f"{'='*60}")

    total = len(results)
    success = sum(1 for r in results if r['status'] == 'SUCCESS')
    failed = sum(1 for r in results if r['status'] == 'FAILED')
    errors = sum(1 for r in results if r['status'] == 'ERROR')

    print(f"Total games processed:  {total}")
    print(f"✓ Successful extractions: {success} ({success/total*100:.1f}%)")
    print(f"✗ Failed extractions:     {failed} ({failed/total*100:.1f}%)")
    print(f"⚠ Errors:                 {errors} ({errors/total*100:.1f}%)")
    print(f"{'='*60}\n")


def main():
    """Main CLI application"""
    print_banner()

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Scrape video stream URLs from Fubo TV NFL recordings',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          # Run with defaults from .env
  python main.py --headless               # Run in headless mode
  python main.py --limit 5                # Test with first 5 games
  python main.py --test-download          # Test ffmpeg download of first stream
  python main.py --verbose                # Enable verbose logging
  python main.py --output custom.csv      # Custom output filename
  python main.py --url "https://..."      # Use custom recordings URL
        """
    )

    parser.add_argument(
        '--headless',
        action='store_true',
        help='Run browser in headless mode (no GUI)'
    )

    parser.add_argument(
        '--no-headless',
        action='store_true',
        help='Explicitly show browser window (overrides --headless and config)'
    )

    parser.add_argument(
        '--limit',
        type=int,
        metavar='N',
        help='Limit number of games to process (useful for testing)'
    )

    parser.add_argument(
        '--test-download',
        action='store_true',
        help='Test download first stream URL with ffmpeg (30 seconds)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose debug logging'
    )

    parser.add_argument(
        '--output',
        type=str,
        metavar='FILE',
        help='Custom output CSV filename (default: auto-generated with timestamp)'
    )

    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from existing CSV (skip already processed games)'
    )

    parser.add_argument(
        '--url',
        type=str,
        metavar='URL',
        help='Custom Fubo recordings URL (overrides .env config)'
    )

    args = parser.parse_args()

    # Validate configuration
    try:
        Config.validate()
    except ValueError as e:
        print(f"Configuration Error: {e}")
        print("\nPlease ensure you have:")
        print("1. Created a .env file (copy from .env.example)")
        print("2. Set FUBO_USERNAME and FUBO_PASSWORD in .env")
        sys.exit(1)

    # Determine headless mode
    # --no-headless explicitly shows the window (for debugging)
    # --headless explicitly hides the window
    # Otherwise, use config default
    if args.no_headless:
        headless = False  # Explicitly show window
    elif args.headless:
        headless = True  # Explicitly hide window
    else:
        headless = Config.HEADLESS_MODE  # Use config default

    # Determine recordings URL
    # --url argument overrides config default
    recordings_url = args.url if args.url else Config.FUBO_RECORDINGS_URL

    print(f"Configuration:")
    print(f"  Username: {Config.FUBO_USERNAME}")
    print(f"  Recordings URL: {recordings_url}")
    print(f"  Headless mode: {headless}")
    print(f"  Delay between games: {Config.DELAY_BETWEEN_GAMES}s")
    if args.limit:
        print(f"  Limit: {args.limit} games")
    print()

    # Initialize scraper
    scraper = None

    try:
        print("Initializing scraper...")
        scraper = FuboScraper(
            username=Config.FUBO_USERNAME,
            password=Config.FUBO_PASSWORD,
            recordings_url=recordings_url,
            headless=headless
        )

        # Login
        print("\n[1/3] Logging in to Fubo TV...")
        scraper.login()
        print("✓ Login successful")

        # Navigate to recordings
        print("\n[2/3] Navigating to NFL recordings...")
        scraper.navigate_to_recordings()
        print("✓ Navigation successful")

        # Scrape games
        print(f"\n[3/3] Extracting game list and stream URLs...")
        games_data = scraper.scrape_all_games(limit=args.limit)

        if not games_data:
            print("\n✗ No games found or extracted. Check the logs for details.")
            sys.exit(1)

        # Print summary
        print_summary(games_data)

        # Export to CSV
        print("Exporting results to CSV...")
        output_file = scraper.export_to_csv(games_data, filename=args.output)
        print(f"✓ Results saved to: {output_file}")

        # Test download if requested
        if args.test_download and games_data:
            # Find first successful stream
            first_stream = next((g for g in games_data if g['status'] == 'SUCCESS'), None)

            if first_stream:
                test_output = os.path.join(Config.OUTPUT_DIR, "test_download.mp4")
                test_ffmpeg_download(
                    stream_url=first_stream['stream_url'],
                    output_file=test_output,
                    duration=30
                )
            else:
                print("\n⚠ No successful streams found for testing download")

        # Final message
        print(f"\n{'='*60}")
        print("COMPLETE!")
        print(f"{'='*60}")
        print(f"CSV file: {output_file}")
        print(f"Log file: {Config.LOG_FILE}")
        print()

    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")

        # Save partial results if any
        if scraper and scraper.processed_games:
            print(f"Saving {len(scraper.processed_games)} partial results...")
            output_file = scraper.export_to_csv(
                scraper.processed_games,
                filename=f"fubo_recordings_partial_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )
            print(f"✓ Partial results saved to: {output_file}")

        sys.exit(130)

    except Exception as e:
        print(f"\n✗ Fatal error: {e}")

        # Save partial results if any
        if scraper and scraper.processed_games:
            print(f"Saving {len(scraper.processed_games)} partial results...")
            try:
                output_file = scraper.export_to_csv(
                    scraper.processed_games,
                    filename=f"fubo_recordings_partial_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                )
                print(f"✓ Partial results saved to: {output_file}")
            except:
                pass

        import traceback
        traceback.print_exc()
        sys.exit(1)

    finally:
        # Cleanup
        if scraper:
            print("\nCleaning up...")
            scraper.cleanup()


if __name__ == "__main__":
    main()
