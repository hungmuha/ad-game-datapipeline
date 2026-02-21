"""
Fubo TV NFL Recordings Scraper
Extracts video stream URLs from Fubo TV NFL recordings
"""
import logging
from selenium.webdriver.remote.webelement import WebElement
import time
import re
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional
from tqdm import tqdm

from seleniumwire import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import TimeoutException, NoSuchElementException, ElementClickInterceptedException, StaleElementReferenceException
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service

from config import Config


class FuboScraper:
    """Scraper for extracting NFL game stream URLs from Fubo TV"""

    def __init__(self, username: str, password: str, recordings_url: str, headless: bool = True):
        """
        Initialize the Fubo scraper

        Args:
            username: Fubo TV account username
            password: Fubo TV account password
            recordings_url: URL to NFL recordings page
            headless: Run browser in headless mode
        """
        self.username = username
        self.password = password
        self.recordings_url = recordings_url
        self.headless = headless
        self.driver = None
        self.logger = self._setup_logger()
        self.processed_games = []

        # Initialize the driver
        self._init_driver()

    def _setup_logger(self) -> logging.Logger:
        """Set up logging configuration"""
        logger = logging.getLogger('FuboScraper')
        logger.setLevel(logging.INFO)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_format)

        # File handler
        file_handler = logging.FileHandler(Config.LOG_FILE)
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_format)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

        return logger

    def _init_driver(self):
        """Initialize selenium-wire Chrome driver"""
        self.logger.info("Initializing Chrome driver with selenium-wire...")

        # Chrome options
        chrome_options = webdriver.ChromeOptions()

        if self.headless:
            chrome_options.add_argument('--headless=new')

        chrome_options.add_argument(f'user-agent={Config.USER_AGENT}')
        chrome_options.add_argument('--disable-blink-features=AutomationControlled')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--window-size=1920,1080')
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)

        # Selenium-wire options for request interception
        seleniumwire_options = {
            'disable_encoding': True  # Ask the server not to compress the response
        }

        # Initialize driver
        try:
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(
                service=service,
                options=chrome_options,
                seleniumwire_options=seleniumwire_options
            )
            self.driver.implicitly_wait(Config.IMPLICIT_WAIT)
            self.logger.info("Chrome driver initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Chrome driver: {e}")
            raise

    def login(self):
        """Log in to Fubo TV"""
        self.logger.info("Attempting to log in to Fubo TV...")

        try:
            # Navigate to Fubo login page
            self.driver.get('https://www.fubo.tv/welcome')
            time.sleep(2)

            # Click sign in button
            try:
                sign_in_btn = WebDriverWait(self.driver, Config.EXPLICIT_WAIT).until(
                    
                    EC.element_to_be_clickable((By.XPATH, "//a[contains(@data-testid, 'sign-in-button')]"))
                )
                sign_in_btn.click()
                time.sleep(2)
            except TimeoutException:
                # Maybe we're already on the login page
                self.logger.info("Sign in button not found, may already be on login page")

            # Enter email
            email_input = WebDriverWait(self.driver, Config.EXPLICIT_WAIT).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='email'], input[name='email'], input[placeholder*='mail' i]"))
            )
            email_input.clear()
            email_input.send_keys(self.username)
            time.sleep(1)

            # Enter password
            password_input = self.driver.find_element(By.CSS_SELECTOR, "input[type='password']")
            password_input.clear()
            password_input.send_keys(self.password)
            time.sleep(1)

            # Submit form
            password_input.send_keys(Keys.RETURN)
            time.sleep(3)

            # Wait for successful login - check for user menu or profile
            try:
                WebDriverWait(self.driver, Config.EXPLICIT_WAIT).until(
                    EC.presence_of_element_located((By.XPATH, "//*[contains(@class, 'user') or contains(@class, 'profile') or contains(@class, 'account')]"))
                )
                self.logger.info("Successfully logged in to Fubo TV")

                time.sleep(3)

                # need to select profile from the list of profiles
                profile_button = WebDriverWait(self.driver, Config.EXPLICIT_WAIT).until(
                    EC.presence_of_element_located((By.XPATH, "//button[contains(@aria-label, 'Hung')]"))
                )
                profile_button.click()
                time.sleep(3)
                # it seems that after this is click the page get redirected to the home page but it seems to not like the https
                
            except TimeoutException:
                self.logger.warning("Could not confirm login success, proceeding anyway...")



        except Exception as e:
            self.logger.error(f"Login failed: {e}")
            raise

    def navigate_to_recordings(self):
        """Navigate to NFL recordings page"""
        self.logger.info(f"Navigating to recordings: {self.recordings_url}")

        try:
            self.driver.get(self.recordings_url)
            time.sleep(3)

            # Wait for page to load - look for recordings container
            try:
                WebDriverWait(self.driver, Config.EXPLICIT_WAIT).until(
                    EC.presence_of_element_located((By.XPATH, "//*[contains(text(), 'All Recordings') or contains(text(), 'Recordings')]"))
                )
                self.logger.info("Successfully navigated to recordings page")
            except TimeoutException:
                self.logger.warning("Could not find 'All Recordings' heading, but page loaded")

        except Exception as e:
            self.logger.error(f"Failed to navigate to recordings: {e}")
            raise

    def extract_game_list(self) -> List[Dict]:
        """
        Extract list of all game recordings from the page

        Returns:
            List of dictionaries containing game metadata
        """
        self.logger.info("Extracting game list...")

        games = []
        seen_games = set()  # Track unique games by title

        try:
            # Wait for initial content to load
            time.sleep(3)
            
            # Try to get total game count from metadata element
            expected_count = None
            try:
                metadata_element = self.driver.find_element(By.CSS_SELECTOR, ".metadata-stack-subtitle")
                metadata_text = metadata_element.text
                # Extract number from text like "24 games"
                import re
                match = re.search(r'(\d+)\s+game', metadata_text, re.IGNORECASE)
                if match:
                    expected_count = int(match.group(1))
                    self.logger.info(f"Found metadata indicating {expected_count} total games")
            except:
                self.logger.warning("Could not find game count metadata, will scroll until no new games appear")
            
            # Scroll and collect games (virtual scrolling removes off-screen elements)
            scroll_attempts = 0
            max_scrolls = 50  # Increase max scrolls for virtual scrolling
            no_new_games_count = 0
            
            while scroll_attempts < max_scrolls:
                # Get currently visible game cards
                current_cards = self.driver.find_elements(By.CSS_SELECTOR, "div.card-root")
                
                # Extract data from visible cards before they disappear
                new_games_found = 0
                for idx, element in enumerate(current_cards):
                    try:
                        # Extract title for uniqueness check
                        title = ""
                        try:
                            title_element = element.find_element(By.CSS_SELECTOR, ".card-stacked-title")
                            title = title_element.text.strip()
                        except:
                            try:
                                title_element = element.find_element(By.XPATH, ".//h3 | .//h4 | .//*[contains(@class, 'title')]")
                                title = title_element.text.strip()
                            except:
                                title = element.text.split('\n')[0] if element.text else ""
                        
                        # Skip if we've seen this game already
                        if title and title not in seen_games:
                            seen_games.add(title)
                            new_games_found += 1
                            
                            # Store full game data
                            game_data = {
                                'index': len(games),
                                'element': element,
                                'title': title,
                                'air_date': '',
                                'network': ''
                            }
                            
                            # Extract air date
                            try:
                                footer_element = element.find_element(By.CSS_SELECTOR, ".card-stacked-footer")
                                footer_text = footer_element.text.strip()
                                if 'Aired:' in footer_text or footer_text.startswith(('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec')):
                                    game_data['air_date'] = footer_text.replace('Aired:', '').strip()
                            except:
                                pass
                            
                            # Extract network - look for card-stacked-channel-logo img
                            try:
                                network_img = element.find_element(By.CSS_SELECTOR, ".card-stacked-channel-logo img")
                                # Try alt attribute first, then extract from src URL
                                network = network_img.get_attribute('alt')
                                if not network or network.strip() == '':
                                    # Extract from src URL (e.g., "wivb_m.png" -> "WIVB")
                                    src = network_img.get_attribute('src') or ''
                                    if src:
                                        # Extract station code from URL
                                        match = re.search(r'station_logos/(\w+)_', src)
                                        if match:
                                            network = match.group(1).upper()
                                game_data['network'] = network if network else ''
                            except:
                                # Fallback: Try to find network text
                                try:
                                    for network in ['ESPN', 'CBS', 'NBC', 'FOX', 'ABC', 'NFL Network', 'WIVB', 'WUTV']:
                                        if network.upper() in element.text.upper():
                                            game_data['network'] = network
                                            break
                                except:
                                    pass
                            
                            games.append(game_data)
                    except:
                        continue
                
                if new_games_found > 0:
                    self.logger.info(f"Scroll {scroll_attempts + 1}: Found {new_games_found} new games (total: {len(games)})")
                    no_new_games_count = 0
                else:
                    no_new_games_count += 1
                
                # Stop if we found all expected games or no new games for 3 attempts
                if expected_count and len(games) >= expected_count:
                    self.logger.info(f"Found all {expected_count} games")
                    break
                if no_new_games_count >= 3:
                    self.logger.info(f"No new games found after {no_new_games_count} scroll attempts")
                    break
                
                # Scroll down
                self.driver.execute_script("window.scrollBy(0, 500);")
                time.sleep(1.5)
                scroll_attempts += 1

            self.logger.info(f"Successfully extracted {len(games)} games")

        except Exception as e:
            self.logger.error(f"Failed to extract game list: {e}")

        return games

    def capture_stream_url(self, game_index: int, game_title: str = None) -> Optional[str]:
        """
        Click on a game and capture the .m3u8 stream URL from network traffic

        Args:
            game_index: Index of the game in the list (0-based)
            game_title: Optional title for logging

        Returns:
            Stream URL if found, None otherwise
        """
        stream_url = None

        try:
            # Clear previous requests
            del self.driver.requests

            # With virtual scrolling, we need to find the element by title, not index
            # Scroll through the list to find the game with matching title
            game_element = None
            max_scroll_attempts = 30
            scroll_attempt = 0
            
            while scroll_attempt < max_scroll_attempts and not game_element:
                # Get currently visible game cards
                game_elements = self.driver.find_elements(By.CSS_SELECTOR, "div.card-root")
                
                # Search for matching title in visible elements
                for element in game_elements:
                    try:
                        title_element = element.find_element(By.CSS_SELECTOR, ".card-stacked-title")
                        element_title = title_element.text.strip()
                        
                        if game_title and element_title == game_title:
                            game_element = element
                            self.logger.debug(f"Found game element: {game_title}")
                            break
                    except:
                        continue
                
                if game_element:
                    break
                
                # Scroll down to load more games
                self.driver.execute_script("window.scrollBy(0, 300);")
                time.sleep(0.5)
                scroll_attempt += 1
            
            if not game_element:
                self.logger.warning(f"Could not find game element for: {game_title}")
                return None
            
            # Scroll element into view
            self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", game_element)
            time.sleep(1)

            # Click the game card
            try:
                game_element.click()
            except (ElementClickInterceptedException, StaleElementReferenceException):
                # Try JavaScript click if regular click fails
                self.driver.execute_script("arguments[0].click();", game_element)

            game_info = f"{game_index + 1}" + (f": {game_title}" if game_title else "")
            self.logger.info(f"Clicked game {game_info}, waiting for stream to load...")

            # Wait for video to initialize
            time.sleep(3)

            # Monitor network requests for .m3u8 URLs
            for request in self.driver.requests:
                if request.response:
                    url = request.url

                    # Check if URL matches stream patterns
                    is_stream = any(pattern in url for pattern in Config.STREAM_URL_PATTERNS)
                    is_fubo_cdn = any(cdn in url.lower() for cdn in Config.CDN_PATTERNS)

                    if is_stream and (is_fubo_cdn or '.m3u8' in url):
                        # Prefer master playlists
                        if 'master' in url or 'finite' in url or stream_url is None:
                            stream_url = url
                            self.logger.info(f"Found stream URL: {url[:100]}...")

                            # If we found a master playlist, we can stop looking
                            if 'master' in url or 'finite' in url:
                                break

            # Close the player - press ESC or click back
            # try:
            #     close_btn = self.driver.find_element(By.XPATH, "//button[contains(@aria-label, 'Close player')]")
            #     close_btn.click()
            #     time.sleep(3)
            # except:
            #     self.driver.back()
            #     time.sleep(3)
                    
            # Navigate back to recordings page
            try:
                self.driver.back()
                time.sleep(3)
                
                # Wait for page to reload and verify we're back on recordings page
                # This ensures elements are fresh for the next iteration
                try:
                    WebDriverWait(self.driver, Config.EXPLICIT_WAIT).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, "div.card-root"))
                    )
                    self.logger.debug("Successfully returned to recordings page")
                except TimeoutException:
                    self.logger.warning("Could not verify return to recordings page, but continuing...")
                    
            except Exception as e:
                self.logger.warning(f"Error navigating back: {e}")

        except Exception as e:
            self.logger.error(f"Error capturing stream URL for game {game_index + 1}: {e}")

            # Try to recover by pressing ESC or going back
            try:
                self.driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.ESCAPE)
                time.sleep(1)
            except:
                try:
                    self.driver.back()
                    time.sleep(2)
                except:
                    pass

        return stream_url

    def scrape_all_games(self, limit: Optional[int] = None) -> List[Dict]:
        """
        Scrape stream URLs from all games

        Args:
            limit: Optional limit on number of games to process

        Returns:
            List of dictionaries with game data and stream URLs
        """
        results = []

        # Get list of all games
        games = self.extract_game_list()

        if not games:
            self.logger.error("No games found on page")
            return results

        # Scroll back to top of the list before processing
        self.driver.execute_script("window.scrollTo(0, 0);")
        time.sleep(2)
        self.logger.info("Scrolled back to top of list")

        # Apply limit if specified
        if limit:
            games = games[:limit]
            self.logger.info(f"Processing limited to first {limit} games")

        self.logger.info(f"Starting to process {len(games)} games...")

        # Process each game with progress bar
        for game in tqdm(games, desc="Processing games", unit="game"):
            try:
                game_index = game['index']
                self.logger.info(f"Processing game {game_index + 1}/{len(games)}: {game['title']}")

                # Capture stream URL - pass index instead of element to avoid stale references
                stream_url = self.capture_stream_url(game_index, game['title'])

                # Record result
                result = {
                    'title': game['title'],
                    'air_date': game['air_date'],
                    'network': game['network'],
                    'stream_url': stream_url if stream_url else 'NOT FOUND',
                    'extraction_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'status': 'SUCCESS' if stream_url else 'FAILED'
                }

                results.append(result)
                self.processed_games.append(result)

                # Log result
                if stream_url:
                    self.logger.info(f"✓ Successfully captured stream for: {game['title']}")
                else:
                    self.logger.warning(f"✗ Failed to capture stream for: {game['title']}")

                # Delay between games
                time.sleep(Config.DELAY_BETWEEN_GAMES)

            except Exception as e:
                self.logger.error(f"Error processing game {game_index + 1}: {e}")

                # Record failed result
                result = {
                    'title': game.get('title', f"Game {game_index + 1}"),
                    'air_date': game.get('air_date', ''),
                    'network': game.get('network', ''),
                    'stream_url': 'ERROR',
                    'extraction_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'status': 'ERROR',
                    'error': str(e)
                }
                results.append(result)

                # Continue with next game
                continue

        self.logger.info(f"Completed processing {len(results)} games")
        success_count = sum(1 for r in results if r['status'] == 'SUCCESS')
        self.logger.info(f"Success rate: {success_count}/{len(results)} ({success_count/len(results)*100:.1f}%)")

        return results

    def export_to_csv(self, data: List[Dict], filename: Optional[str] = None) -> str:
        """
        Export scraped data to CSV file

        Args:
            data: List of dictionaries with game data
            filename: Optional custom filename

        Returns:
            Path to exported CSV file
        """
        if not data:
            self.logger.warning("No data to export")
            return ""

        # Generate filename if not provided
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"fubo_recordings_{timestamp}.csv"

        filepath = os.path.join(Config.OUTPUT_DIR, filename)

        # Create DataFrame
        df = pd.DataFrame(data)

        # Reorder columns
        column_order = ['title', 'air_date', 'network', 'stream_url', 'status', 'extraction_time']
        if 'error' in df.columns:
            column_order.append('error')

        df = df[column_order]

        # Sort by air date (most recent first) if possible
        # Note: This assumes air_date is in a sortable format
        try:
            df = df.sort_values('air_date', ascending=False)
        except:
            pass

        # Export to CSV
        df.to_csv(filepath, index=False, encoding='utf-8')

        self.logger.info(f"Exported {len(data)} records to: {filepath}")

        return filepath

    def cleanup(self):
        """Clean up resources and close browser"""
        if self.driver:
            try:
                self.driver.quit()
                self.logger.info("Browser closed successfully")
            except Exception as e:
                self.logger.error(f"Error closing browser: {e}")


# Import os for file operations
import os
