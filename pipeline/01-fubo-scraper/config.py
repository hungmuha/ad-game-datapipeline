"""
Configuration settings for Fubo NFL Scraper
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Application configuration"""

    # Fubo TV Credentials
    FUBO_USERNAME = os.getenv('FUBO_USERNAME')
    FUBO_PASSWORD = os.getenv('FUBO_PASSWORD')
    FUBO_RECORDINGS_URL = os.getenv('FUBO_RECORDINGS_URL',
                                     'https://www.fubo.tv/p/my-stuff/recordings/leagueId/191277')

    # Browser Settings
    HEADLESS_MODE = os.getenv('HEADLESS_MODE', 'True').lower() == 'true'

    # Scraper Settings
    DELAY_BETWEEN_GAMES = int(os.getenv('DELAY_BETWEEN_GAMES', '2'))
    IMPLICIT_WAIT = int(os.getenv('IMPLICIT_WAIT', '10'))
    EXPLICIT_WAIT = int(os.getenv('EXPLICIT_WAIT', '15'))

    # Paths
    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
    LOG_FILE = os.path.join(OUTPUT_DIR, 'scraper.log')

    # User Agent (to avoid bot detection)
    USER_AGENT = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'

    # Network request patterns to look for
    STREAM_URL_PATTERNS = [
        'finite.m3u8',
        'media.m3u8',
        'playlist.m3u8',
        'master.m3u8',
        '.mpd'
    ]

    # Fubo CDN patterns
    CDN_PATTERNS = [
        'fubo',
        'playlist',
        'stream',
        'cdn'
    ]

    @classmethod
    def validate(cls):
        """Validate required configuration"""
        if not cls.FUBO_USERNAME:
            raise ValueError("FUBO_USERNAME not set in .env file")
        if not cls.FUBO_PASSWORD:
            raise ValueError("FUBO_PASSWORD not set in .env file")

        # Ensure output directory exists
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)

        return True
