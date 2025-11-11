import time
import csv
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from selenium.webdriver.chrome.options import Options
import random
import requests
from typing import Dict, Optional
import json
import os
from mistralai import Mistral

# ============= CONFIGURATION =============
APOLLO_API_KEY = "hX73XMlw-UItMQxHU...."  # Replace with your actual Apollo API key
APOLLO_RATE_LIMIT_DELAY = 1  # seconds between API calls
MISTRAL_API_KEY = ""  # Replace with your Mistral API key or set via environment variable

# ✅ NEW: Flexible scraping configuration
SCRAPING_CONFIG = {
    'default_posts_per_url': 10,
    'max_scroll_attempts': 10,
    'scroll_delay_min': 2,
    'scroll_delay_max': 4,
    'post_extraction_delay_min': 0.5,
    'post_extraction_delay_max': 1.5,
    'url_delay_min': 5,
    'url_delay_max': 10
}

# ==========================================
class MistralPostClassifier:
    """Mistral AI classifier for filtering agency vs individual hiring posts"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = Mistral(api_key=api_key)
        self.classifications_made = 0
        self.posts_kept = 0
        self.posts_removed = 0

    def classify_post(self, post_text: str) -> str:
        """
        Classify whether a post is looking for agency partnership or hiring individuals.
        Returns: 'KEEP' for agency posts, 'REMOVE' for individual hiring posts
        """

        prompt = f"""You are a LinkedIn post classifier. Your task is to determine if a post is:
1. Looking for AGENCY/PARTNER collaboration (marketing agency, recruitment agency, consulting firm, etc.) - label as "KEEP"
2. Hiring INDIVIDUAL employees/freelancers/specialists for specific roles - label as "REMOVE"

Rules:
- If the post is seeking an agency, partner, consultant firm, or service provider → "KEEP"
- If the post is hiring individual people for jobs (manager, specialist, developer, etc.) → "REMOVE"
- If the post mentions "we're hiring", "looking for candidates", "join our team" for individual roles → "REMOVE"
- If unclear, default to "REMOVE"

Post text:
{post_text}

Respond with ONLY ONE WORD: either "KEEP" or "REMOVE"
"""

        try:
            response = self.client.chat.complete(
                model="mistral-small",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,  # Low temperature for consistent classification
                max_tokens=10
            )

            result = response.choices[0].message.content.strip().upper()

            # Ensure we only get KEEP or REMOVE
            if "KEEP" in result:
                self.posts_kept += 1
                return "KEEP"
            elif "REMOVE" in result:
                self.posts_removed += 1
                return "REMOVE"
            else:
                self.posts_removed += 1
                return "REMOVE"  # Default to REMOVE if unclear

        except Exception as e:
            print(f"   ⚠️ Error classifying post: {e}")
            self.posts_removed += 1
            return "REMOVE"  # Default to REMOVE on error

    def filter_posts(self, data: list) -> list:
        """
        Filter list of posts using Mistral classification.
        Returns only posts marked as KEEP.
        """
        print(f"\n{'='*70}")
        print(f"🤖 CLASSIFYING POSTS WITH MISTRAL AI")
        print(f"{'='*70}\n")

        filtered_data = []

        for idx, record in enumerate(data, 1):
            post_text = record.get('post_text', '')
            author_name = record.get('author_name', 'Unknown')

            if not post_text:
                print(f"[{idx}/{len(data)}] {author_name}: REMOVE (no post text)")
                self.posts_removed += 1
                continue

            classification = self.classify_post(post_text)
            self.classifications_made += 1

            preview = post_text[:80].replace('\n', ' ')
            print(f"[{idx}/{len(data)}] {author_name}: {classification}")
            print(f"   Preview: {preview}...")

            if classification == "KEEP":
                filtered_data.append(record)

            # Add small delay to avoid rate limits
            if idx < len(data):
                time.sleep(0.5)

        self.print_summary()
        return filtered_data

    def print_summary(self):
        """Print classification statistics"""
        print(f"\n{'='*70}")
        print(f"🤖 MISTRAL CLASSIFICATION SUMMARY:")
        print(f"{'='*70}")
        print(f"  Total posts classified: {self.classifications_made}")
        print(f"  ✓ Posts KEPT (agency): {self.posts_kept}")
        print(f"  ✗ Posts REMOVED (hiring): {self.posts_removed}")
        if self.classifications_made > 0:
            keep_rate = (self.posts_kept / self.classifications_made) * 100
            print(f"  Keep rate: {keep_rate:.1f}%")
        print(f"{'='*70}\n")


class MistralCompanyAnalyzer:
    """Mistral AI analyzer for extracting company industry and category"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = Mistral(api_key=api_key)
        self.analyses_made = 0

    def extract_company_info(self, company_name: str) -> dict:
        """
        Extracts company description, category, and industry using Mistral AI
        Returns: dict with description, category, and industry
        """
        if not company_name or company_name.strip() == '':
            return {
                'description': 'Data unavailable',
                'category': 'Data unavailable',
                'industry': 'Data unavailable'
            }

        prompt = f"""
You are an expert company researcher.
Provide a concise profile of the company named "{company_name}".

Return the following ONLY:

Description: A single sentence (max 30 words) describing what the company does.
Category: One of [B2B, B2C, B2B2C, SaaS, Marketplace, Enterprise, Consumer, Mixed].
Industry: The main industry or sector (e.g., Fintech, Healthcare, Edtech, AI, Logistics, Energy).

Rules:
- Keep it factual and short.
- Do NOT include explanations, extra text, or commentary.
- Output in this exact format:

Description: <30-word summary>
Category: <one of the above>
Industry: <industry/sector>
"""

        try:
            response = self.client.chat.complete(
                model="mistral-small-latest",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.2,
                max_tokens=120
            )

            output = response.choices[0].message.content.strip()
            self.analyses_made += 1

            # Parse the response
            description = "Data unavailable"
            category = "Data unavailable"
            industry = "Data unavailable"

            import re
            desc_match = re.search(r"Description:\s*(.+)", output)
            cat_match = re.search(r"Category:\s*(.+)", output)
            ind_match = re.search(r"Industry:\s*(.+)", output)

            if desc_match:
                description = desc_match.group(1).strip().split("\n")[0][:200]
            if cat_match:
                category = cat_match.group(1).strip().split("\n")[0]
            if ind_match:
                industry = ind_match.group(1).strip().split("\n")[0]

            return {
                'description': description,
                'category': category,
                'industry': industry
            }

        except Exception as e:
            print(f"   ⚠️ Error analyzing company {company_name}: {e}")
            self.analyses_made += 1
            return {
                'description': 'Data unavailable',
                'category': 'Data unavailable',
                'industry': 'Data unavailable'
            }

    def analyze_companies(self, data: list) -> list:
        """
        Analyze companies and add industry/category information
        Returns updated data list with new fields
        """
        print(f"\n{'='*70}")
        print(f"🏢 ANALYZING COMPANIES WITH MISTRAL AI")
        print(f"{'='*70}\n")

        enriched_data = []

        for idx, record in enumerate(data, 1):
            company_name = record.get('apollo_company', '') or record.get('company_name', '')
            
            if not company_name or company_name.strip() == '':
                print(f"[{idx}/{len(data)}] No company name - skipping analysis")
                record['company_description'] = 'Data unavailable'
                record['company_category'] = 'Data unavailable'
                record['company_industry'] = 'Data unavailable'
                enriched_data.append(record)
                continue

            print(f"[{idx}/{len(data)}] Analyzing: {company_name}")
            
            company_info = self.extract_company_info(company_name)
            
            # Add new fields to the record
            record['company_description'] = company_info['description']
            record['company_category'] = company_info['category']
            record['company_industry'] = company_info['industry']
            
            print(f"   Industry: {company_info['industry']}")
            print(f"   Category: {company_info['category']}")
            
            enriched_data.append(record)

            # Add small delay to avoid rate limits
            if idx < len(data):
                time.sleep(0.5)

        self.print_summary()
        return enriched_data

    def print_summary(self):
        """Print analysis statistics"""
        print(f"\n{'='*70}")
        print(f"🏢 COMPANY ANALYSIS SUMMARY:")
        print(f"{'='*70}")
        print(f"  Total companies analyzed: {self.analyses_made}")
        print(f"{'='*70}\n")


class ApolloEmailEnricher:
    """Apollo.io API client for email enrichment - ENHANCED VERSION with Phone Support"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.apollo.io/v1"
        self.headers = {
            "Content-Type": "application/json",
            "Cache-Control": "no-cache",
            "X-Api-Key": api_key
        }

        self.requests_made = 0
        self.successful_enrichments = 0
        self.failed_enrichments = 0
        self.company_pages_skipped = 0
        self.unavailable_count = 0
        self.phones_found = 0  # ✅ NEW: Track phone numbers found

    def extract_linkedin_id(self, linkedin_url: str) -> Optional[str]:
        """Extract LinkedIn profile ID/username from URL"""
        if not linkedin_url:
            return None

        try:
            parts = linkedin_url.rstrip('/').split('/')
            if 'in' in parts:
                idx = parts.index('in')
                if idx + 1 < len(parts):
                    return parts[idx + 1].split('?')[0]
            return None
        except Exception as e:
            print(f"  ⚠️ Error extracting LinkedIn ID from {linkedin_url}: {e}")
            return None

    def is_company_page(self, linkedin_url: str) -> bool:
        """Check if URL is a company page (not a personal profile)"""
        if not linkedin_url:
            return False
        return '/company/' in linkedin_url

    def get_email_from_linkedin(self, linkedin_url: str, name: str = None) -> Dict[str, any]:
        """
        ✅ ENHANCED: Fetch verified email AND phone from Apollo.io using LinkedIn profile URL
        Returns: dict with email, phone, verification status, and additional data
        """
        if not linkedin_url:
            return {
                'email': '',
                'email_status': 'no_linkedin_url',
                'apollo_confidence': '',
                'phone': '',
                'company_name': '',
                'title': ''
            }

        # Skip company pages - Apollo only works with personal profiles
        if self.is_company_page(linkedin_url):
            self.company_pages_skipped += 1
            print(f"  ⏩ Skipping company page (Apollo only enriches personal profiles)")
            return {
                'email': '',
                'email_status': 'company_page_not_supported',
                'apollo_confidence': '',
                'phone': '',
                'company_name': '',
                'title': ''
            }

        linkedin_id = self.extract_linkedin_id(linkedin_url)
        if not linkedin_id:
            return {
                'email': '',
                'email_status': 'invalid_linkedin_url',
                'apollo_confidence': '',
                'phone': '',
                'company_name': '',
                'title': ''
            }

        self.requests_made += 1
        print(f"  🔍 Apollo API: Looking up {name if name else 'profile'} via LinkedIn")

        try:
            url = f"{self.base_url}/people/match"
            
            payload = {
                "linkedin_url": linkedin_url
            }

            if name:
                payload["name"] = name

            headers = {
                "Content-Type": "application/json",
                "Cache-Control": "no-cache",
                "X-Api-Key": self.api_key
            }

            print(f"  📡 Calling Apollo API...")
            print(f"  🔗 URL: {linkedin_url}")
            
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=15
            )

            print(f"  📊 Response Status: {response.status_code}")

            if response.status_code == 200:
                data = response.json()
                person = data.get('person', {})

                # Extract email
                email = person.get('email', '')
                email_status = person.get('email_status', '')

                # ✅ ENHANCED: Better phone number extraction with multiple fallbacks
                phone = ''
                
                # Method 1: Try phone_numbers array (most reliable)
                phone_numbers = person.get('phone_numbers', [])
                if phone_numbers and len(phone_numbers) > 0:
                    # Try to get the first mobile or direct number
                    for phone_obj in phone_numbers:
                        phone_type = phone_obj.get('type', '').lower()
                        if 'mobile' in phone_type or 'direct' in phone_type:
                            phone = phone_obj.get('sanitized_number', '') or phone_obj.get('raw_number', '')
                            if phone:
                                break
                    
                    # If no mobile/direct found, just get the first number
                    if not phone and phone_numbers:
                        phone = phone_numbers[0].get('sanitized_number', '') or phone_numbers[0].get('raw_number', '')

                # Method 2: Try direct phone fields
                if not phone:
                    phone = (person.get('sanitized_phone') or 
                            person.get('phone') or 
                            person.get('mobile_phone') or
                            person.get('corporate_phone') or '')

                # Method 3: Try nested organization phone
                if not phone and person.get('organization'):
                    phone = person['organization'].get('phone', '')

                # Clean phone number (remove extra spaces, ensure proper format)
                if phone:
                    phone = phone.strip()
                    self.phones_found += 1  # ✅ NEW: Increment phone counter

                # Extract company name
                company_name = ''
                if person.get('organization'):
                    company_name = person['organization'].get('name', '')

                title = person.get('title', '')

                # Better confidence scoring
                confidence = ''
                if email:
                    if email_status == 'verified':
                        confidence = 'high'
                    elif email_status == 'guessed':
                        confidence = 'medium'
                    elif email_status == 'likely':
                        confidence = 'medium-low'
                    elif email_status == 'extrapolated':
                        confidence = 'low'
                    else:
                        confidence = 'unknown'

                result = {
                    'email': email or '',
                    'email_status': email_status or 'not_found',
                    'apollo_confidence': confidence,
                    'phone': phone or '',  # ✅ Ensured phone is included
                    'company_name': company_name,
                    'title': title
                }

                # Better status reporting
                if email:
                    self.successful_enrichments += 1
                    status_icon = "✔" if email_status == 'verified' else "~"
                    email_display = email[:30] + '...' if len(email) > 30 else email
                    print(f"  {status_icon} Found email: {email_display} ({email_status}, confidence: {confidence})")
                else:
                    self.failed_enrichments += 1
                    if email_status == 'unavailable':
                        self.unavailable_count += 1
                        print(f"  ⚠️ No email available in Apollo database")
                    else:
                        print(f"  ✗ No email found (status: {email_status or 'not_found'})")

                # ✅ ENHANCED: Better phone reporting
                if phone:
                    print(f"  📞 Found phone: {phone}")
                else:
                    print(f"  ⚠️ No phone number available")
                
                if company_name:
                    print(f"  ✔ Found company: {company_name}")

                return result

            elif response.status_code == 404:
                self.failed_enrichments += 1
                print(f"  ✗ Profile not found in Apollo database")
                return {
                    'email': '',
                    'email_status': 'not_in_apollo_db',
                    'apollo_confidence': '',
                    'phone': '',
                    'company_name': '',
                    'title': ''
                }

            elif response.status_code == 401:
                self.failed_enrichments += 1
                print(f"  ✗ ❌ Apollo: Invalid API key (401)")
                print(f"  💡 Check your API key: {self.api_key[:10]}...")
                return {
                    'email': '',
                    'email_status': 'auth_error_401',
                    'apollo_confidence': '',
                    'phone': '',
                    'company_name': '',
                    'title': ''
                }

            elif response.status_code == 403:
                self.failed_enrichments += 1
                print(f"  ✗ ❌ Apollo 403 Forbidden - Check API key permissions")
                return {
                    'email': '',
                    'email_status': 'auth_error_403',
                    'apollo_confidence': '',
                    'phone': '',
                    'company_name': '',
                    'title': ''
                }

            elif response.status_code == 429:
                print(f"  ⚠️ Rate limit hit, waiting 10s...")
                time.sleep(10)
                return self.get_email_from_linkedin(linkedin_url, name)

            elif response.status_code == 422:
                self.failed_enrichments += 1
                print(f"  ✗ Invalid request (422) - check LinkedIn URL format")
                try:
                    error_detail = response.json()
                    print(f"  📋 Error details: {error_detail}")
                except:
                    print(f"  📋 Response: {response.text[:200]}")
                return {
                    'email': '',
                    'email_status': 'invalid_request',
                    'apollo_confidence': '',
                    'phone': '',
                    'company_name': '',
                    'title': ''
                }

            elif response.status_code == 400:
                self.failed_enrichments += 1
                print(f"  ✗ Bad Request (400)")
                try:
                    error_detail = response.json()
                    error_message = error_detail.get('error', {}).get('message', str(error_detail))
                    print(f"  📋 Error: {error_message}")
                    
                    if 'credit' in error_message.lower():
                        print(f"  💳 Apollo API credits exhausted!")
                        return {
                            'email': '',
                            'email_status': 'credits_exhausted',
                            'apollo_confidence': '',
                            'phone': '',
                            'company_name': '',
                            'title': ''
                        }
                except:
                    print(f"  📋 Response: {response.text[:300]}")
                
                return {
                    'email': '',
                    'email_status': 'api_error_400',
                    'apollo_confidence': '',
                    'phone': '',
                    'company_name': '',
                    'title': ''
                }

            else:
                self.failed_enrichments += 1
                print(f"  ✗ Apollo API error: {response.status_code}")
                try:
                    error_detail = response.json()
                    print(f"  📋 Error: {error_detail}")
                except:
                    print(f"  📋 Response: {response.text[:200]}")
                return {
                    'email': '',
                    'email_status': f'api_error_{response.status_code}',
                    'apollo_confidence': '',
                    'phone': '',
                    'company_name': '',
                    'title': ''
                }

        except requests.exceptions.Timeout:
            self.failed_enrichments += 1
            print(f"  ✗ Apollo API timeout")
            return {
                'email': '',
                'email_status': 'timeout',
                'apollo_confidence': '',
                'phone': '',
                'company_name': '',
                'title': ''
            }

        except json.JSONDecodeError:
            self.failed_enrichments += 1
            print(f"  ✗ Apollo returned invalid JSON")
            return {
                'email': '',
                'email_status': 'invalid_response',
                'apollo_confidence': '',
                'phone': '',
                'company_name': '',
                'title': ''
            }

        except requests.exceptions.RequestException as e:
            self.failed_enrichments += 1
            print(f"  ✗ Apollo request failed: {str(e)[:100]}")
            return {
                'email': '',
                'email_status': 'request_error',
                'apollo_confidence': '',
                'phone': '',
                'company_name': '',
                'title': ''
            }

        except Exception as e:
            self.failed_enrichments += 1
            print(f"  ✗ Apollo unexpected error: {str(e)[:100]}")
            import traceback
            print(f"  📋 Traceback: {traceback.format_exc()[:300]}")
            return {
                'email': '',
                'email_status': 'error',
                'apollo_confidence': '',
                'phone': '',
                'company_name': '',
                'title': ''
            }

    def print_summary(self):
        """✅ ENHANCED: Print enrichment statistics including phone numbers"""
        print(f"\n{'='*70}")
        print(f"📧 APOLLO ENRICHMENT SUMMARY:")
        print(f"{'='*70}")
        print(f"  Total API calls made: {self.requests_made}")
        print(f"  ✓ Successful enrichments: {self.successful_enrichments}")
        print(f"  ✗ Failed enrichments: {self.failed_enrichments}")
        print(f"  ⊗ Company pages skipped: {self.company_pages_skipped}")
        print(f"  ⚠️ Unavailable in database: {self.unavailable_count}")
        print(f"  📞 Phone numbers found: {self.phones_found}")  # ✅ NEW: Phone stats
        if self.requests_made > 0:
            success_rate = (self.successful_enrichments / self.requests_made) * 100
            phone_rate = (self.phones_found / self.requests_made) * 100
            print(f"  Success rate (email): {success_rate:.1f}%")
            print(f"  Success rate (phone): {phone_rate:.1f}%")  # ✅ NEW
        print(f"{'='*70}\n")


class LinkedInLeadScraper:
    def __init__(self, email, password, apollo_api_key=None, mistral_api_key=None, config=None):
        """Initialize the scraper with LinkedIn credentials and optional API keys"""
        self.email = email
        self.password = password
        self.driver = None
        self.apollo_enricher = ApolloEmailEnricher(apollo_api_key) if apollo_api_key else None
        self.mistral_classifier = MistralPostClassifier(mistral_api_key) if mistral_api_key else None
        self.company_analyzer = MistralCompanyAnalyzer(mistral_api_key) if mistral_api_key else None
        self.config = config or SCRAPING_CONFIG
        self.setup_driver()

    def setup_driver(self):
        """Setup Chrome driver with options"""
        chrome_options = Options()
        # Uncomment the line below if you want headless mode
        # chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

        self.driver = webdriver.Chrome(options=chrome_options)
        self.driver.maximize_window()

    def login(self):
        """Login to LinkedIn"""
        print("Logging into LinkedIn...")
        self.driver.get("https://www.linkedin.com/login")
        time.sleep(2)

        try:
            # Enter email
            email_field = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.ID, "username"))
            )
            email_field.send_keys(self.email)

            # Enter password
            password_field = self.driver.find_element(By.ID, "password")
            password_field.send_keys(self.password)

            # Click login button
            login_button = self.driver.find_element(By.XPATH, "//button[@type='submit']")
            login_button.click()

            time.sleep(5)
            print("Login successful!")
        except Exception as e:
            print(f"Login failed: {e}")
            raise

    def scroll_and_load_posts(self, max_posts=None):
        """Scroll through the page to load posts with infinite scrolling"""
        if max_posts is None:
            max_posts = self.config['default_posts_per_url']

        print(f"Loading posts (target: {max_posts})...")
        last_height = self.driver.execute_script("return document.body.scrollHeight")
        posts_loaded = 0
        scroll_attempts = 0
        max_scroll_attempts = self.config['max_scroll_attempts']

        while posts_loaded < max_posts and scroll_attempts < max_scroll_attempts:
            # Scroll down
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            scroll_delay = random.uniform(
                self.config['scroll_delay_min'],
                self.config['scroll_delay_max']
            )
            time.sleep(scroll_delay)

            # Check how many posts are loaded
            try:
                posts = self.driver.find_elements(By.CSS_SELECTOR, "div.feed-shared-update-v2")
                posts_loaded = len(posts)
                print(f"Posts loaded: {posts_loaded}/{max_posts}")
            except:
                pass

            # Calculate new scroll height and compare with last scroll height
            new_height = self.driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                # Try scrolling up a bit and then down again
                self.driver.execute_script("window.scrollBy(0, -300);")
                time.sleep(1)
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(2)

            last_height = new_height
            scroll_attempts += 1

        print(f"Finished scrolling. Total posts found: {posts_loaded}")
        return min(posts_loaded, max_posts)

    def extract_post_data(self, post_element):
        """Extract data from a single post element"""
        data = {
            'post_text': '',
            'author_name': '',
            'author_profile_url': '',
            'company_name': '',
            'time_posted': '',
            'post_type': '',
            'post_url': ''
        }

        try:
            # Check if it's a company post or personal post
            is_company_post = False

            # Try to find company page link first (for company posts)
            company_selectors = [
                "a[href*='/company/']",
                "a.app-aware-link[href*='/company/']"
            ]

            for selector in company_selectors:
                try:
                    company_link = post_element.find_element(By.CSS_SELECTOR, selector)
                    company_url = company_link.get_attribute('href')
                    if company_url and '/company/' in company_url:
                        data['author_profile_url'] = company_url.split('?')[0]
                        data['post_type'] = 'company_post'
                        is_company_post = True
                        break
                except:
                    continue

            # If not a company post, try personal profile
            if not is_company_post:
                author_selectors = [
                    "a.app-aware-link[href*='/in/']",
                    "a[href*='/in/'][data-control-name*='actor']",
                    "div.update-components-actor a[href*='/in/']",
                    "a.update-components-actor__meta-link"
                ]

                for selector in author_selectors:
                    try:
                        author_link = post_element.find_element(By.CSS_SELECTOR, selector)
                        profile_url = author_link.get_attribute('href')
                        if profile_url and '/in/' in profile_url:
                            data['author_profile_url'] = profile_url.split('?')[0]
                            data['post_type'] = 'personal_post'
                            break
                    except:
                        continue

            # Extract author/company name - with duplicate removal
            name_selectors = [
                "span.update-components-actor__name span[aria-hidden='true']",
                "span.update-components-actor__name",
                "div.update-components-actor__container span[dir='ltr']",
                "a.update-components-actor__meta-link span[aria-hidden='true']"
            ]

            for selector in name_selectors:
                try:
                    author_name_element = post_element.find_element(By.CSS_SELECTOR, selector)
                    name = author_name_element.text.strip()

                    # Remove duplicate names
                    if '\n' in name:
                        name_parts = name.split('\n')
                        unique_parts = []
                        for part in name_parts:
                            if part not in unique_parts:
                                unique_parts.append(part)
                        name = unique_parts[0] if len(unique_parts) == 1 else ' '.join(unique_parts)

                    if name and len(name) > 2:
                        data['author_name'] = name
                        break
                except:
                    continue

            # Extract company name
            company_selectors = [
                "span.update-components-actor__description",
                "div.update-components-actor__description",
                "span.update-components-actor__meta"
            ]

            for selector in company_selectors:
                try:
                    company_element = post_element.find_element(By.CSS_SELECTOR, selector)
                    company = company_element.text.strip()
                    if company and company != data['author_name']:
                        data['company_name'] = company
                        break
                except:
                    continue

            # Extract time posted
            time_selectors = [
                "span.update-components-actor__sub-description span[aria-hidden='true']",
                "span.update-components-actor__sub-description",
                "div.update-components-actor time"
            ]

            for selector in time_selectors:
                try:
                    time_element = post_element.find_element(By.CSS_SELECTOR, selector)
                    time_text = time_element.text.strip()
                    if time_text:
                        data['time_posted'] = time_text
                        break
                except:
                    continue

            # Extract post text - try to click "see more" first
            try:
                see_more_buttons = post_element.find_elements(By.CSS_SELECTOR, "button.feed-shared-inline-show-more-text__see-more-less-toggle, button[aria-label*='see more']")
                if see_more_buttons:
                    self.driver.execute_script("arguments[0].click();", see_more_buttons[0])
                    time.sleep(0.5)
            except:
                pass

            # Extract full post text
            text_selectors = [
                "div.feed-shared-update-v2__description span[dir='ltr']",
                "div.update-components-text span[dir='ltr']",
                "div.feed-shared-text span[dir='ltr']",
                "div.feed-shared-update-v2__description",
                "div.update-components-text"
            ]

            for selector in text_selectors:
                try:
                    post_text_element = post_element.find_element(By.CSS_SELECTOR, selector)
                    text = post_text_element.text.strip()
                    if text and len(text) > 10:
                        data['post_text'] = text
                        break
                except:
                    continue

            # Extract post URL
            post_url_selectors = [
                "a[data-attribute-name='content']",
                "a.app-aware-link[href*='/feed/update/']",
                "a[href*='/posts/'][href*='activity']",
                "div.feed-shared-update-v2 a[href*='activity-']"
            ]

            for selector in post_url_selectors:
                try:
                    post_link = post_element.find_element(By.CSS_SELECTOR, selector)
                    post_url = post_link.get_attribute('href')
                    if post_url and ('activity' in post_url or 'feed/update' in post_url):
                        data['post_url'] = post_url.split('?')[0]
                        break
                except:
                    continue
            
            # Fallback: Try to get post ID from the post element itself
            if not data['post_url']:
                try:
                    post_id = post_element.get_attribute('data-urn')
                    if post_id:
                        if 'activity:' in post_id:
                            activity_id = post_id.split('activity:')[-1].split(',')[0]
                            data['post_url'] = f"https://www.linkedin.com/feed/update/urn:li:activity:{activity_id}/"
                except:
                    pass

        except Exception as e:
            print(f"Error extracting post data: {e}")

        return data

    def scrape_search_url(self, url, max_posts=None):
        """Scrape posts from a search URL"""
        if max_posts is None:
            max_posts = self.config['default_posts_per_url']

        print(f"\nScraping URL: {url}")
        print(f"Target posts: {max_posts}")

        self.driver.get(url)
        time.sleep(5)

        self.scroll_and_load_posts(max_posts)

        posts = self.driver.find_elements(By.CSS_SELECTOR, "div.feed-shared-update-v2")

        scraped_data = []
        for i, post in enumerate(posts[:max_posts]):
            print(f"Scraping post {i+1}/{min(len(posts), max_posts)}")
            post_data = self.extract_post_data(post)

            if post_data['author_name'] or post_data['post_text']:
                post_data['search_url'] = url
                scraped_data.append(post_data)

            extraction_delay = random.uniform(
                self.config['post_extraction_delay_min'],
                self.config['post_extraction_delay_max']
            )
            time.sleep(extraction_delay)

        print(f"Scraped {len(scraped_data)} posts from this URL")
        return scraped_data

    def scrape_multiple_urls(self, urls, posts_per_url=None):
        """Scrape multiple search URLs"""
        all_data = []

        for i, url in enumerate(urls):
            print(f"\n{'='*60}")
            print(f"Processing URL {i+1}/{len(urls)}")
            print(f"{'='*60}")

            if posts_per_url is None:
                max_posts = self.config['default_posts_per_url']
            elif isinstance(posts_per_url, int):
                max_posts = posts_per_url
            elif isinstance(posts_per_url, dict):
                max_posts = posts_per_url.get(url, self.config['default_posts_per_url'])
            elif isinstance(posts_per_url, list):
                max_posts = posts_per_url[i] if i < len(posts_per_url) else self.config['default_posts_per_url']
            else:
                max_posts = self.config['default_posts_per_url']

            try:
                data = self.scrape_search_url(url, max_posts)
                all_data.extend(data)

                if i < len(urls) - 1:
                    delay = random.uniform(
                        self.config['url_delay_min'],
                        self.config['url_delay_max']
                    )
                    print(f"Waiting {delay:.1f} seconds before next URL...")
                    time.sleep(delay)

            except Exception as e:
                print(f"Error scraping URL {url}: {e}")
                continue

        return all_data

    def enrich_with_apollo(self, data):
        """✅ ENHANCED: Enrich scraped data with Apollo.io emails AND phones"""
        if not self.apollo_enricher:
            print("\n⚠️ Apollo API key not provided. Skipping email enrichment.")
            return data

        print(f"\n{'='*70}")
        print(f"📧 ENRICHING {len(data)} PROFILES WITH APOLLO.IO")
        print(f"{'='*70}\n")

        enriched_data = []

        for idx, record in enumerate(data, 1):
            post_type = record.get('post_type', 'unknown')
            author_name = record.get('author_name', 'Unknown')
            linkedin_url = record.get('author_profile_url', '')

            print(f"[{idx}/{len(data)}] {author_name} ({post_type})")

            email_data = self.apollo_enricher.get_email_from_linkedin(
                linkedin_url,
                author_name
            )

            # ✅ FIXED: Properly map ALL Apollo data including phone
            enriched_record = {
                'author_name': author_name,
                'author_profile_url': linkedin_url,
                'post_type': post_type,
                'verified_email': email_data['email'],
                'email_verification_status': email_data['email_status'],
                'apollo_confidence': email_data['apollo_confidence'],
                'apollo_phone': email_data['phone'],  # ✅ CRITICAL: Phone number mapping
                'apollo_company': email_data['company_name'],
                'apollo_title': email_data['title'],
                'company_name': record.get('company_name', ''),
                'time_posted': record.get('time_posted', ''),
                'post_text': record.get('post_text', ''),
                'post_url': record.get('post_url', ''),
                'search_url': record.get('search_url', '')
            }

            enriched_data.append(enriched_record)

            if idx < len(data):
                time.sleep(APOLLO_RATE_LIMIT_DELAY)

        self.apollo_enricher.print_summary()

        return enriched_data

    def save_to_csv(self, data, enrich_with_email=False, filter_with_mistral=False, analyze_companies=False):
        """✅ FIXED: Save scraped data to CSV with proper phone number column"""
        if not data:
            print("No data to save!")
            return None

        # Filter with Mistral AI first if enabled
        if filter_with_mistral and self.mistral_classifier:
            data = self.mistral_classifier.filter_posts(data)

            if not data:
                print("\n⚠️ No posts remaining after Mistral filtering!")
                return None

        # Enrich with Apollo emails and phones
        if enrich_with_email and self.apollo_enricher:
            data = self.enrich_with_apollo(data)

        # Analyze companies for industry/category if enabled
        if analyze_companies and self.company_analyzer:
            data = self.company_analyzer.analyze_companies(data)

        # Generate unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"linkedin_leads_{timestamp}.csv"

        # ✅ FIXED: Define CSV headers with phone number included
        headers = ['author_name', 'author_profile_url', 'post_type', 'company_name', 'time_posted', 'post_text', 'post_url', 'search_url']
        
        if enrich_with_email and self.apollo_enricher:
            headers = [
                'author_name',
                'author_profile_url',
                'post_type',
                'verified_email',
                'email_verification_status',
                'apollo_confidence',
                'apollo_phone',  # ✅ CRITICAL: Phone column added to headers
                'apollo_company',
                'apollo_title',
                'company_name',
                'time_posted',
                'post_text',
                'post_url',
                'search_url'
            ]
        
        # Add company analysis columns if enabled
        if analyze_companies and self.company_analyzer:
            headers.extend(['company_description', 'company_category', 'company_industry'])

        # Write to CSV
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(data)

        print(f"\n{'='*70}")
        print(f"✅ RESULTS SAVED SUCCESSFULLY!")
        print(f"{'='*70}")
        print(f"📊 Total records: {len(data)}")
        print(f"📁 Output file: {filename}")

        # ✅ ENHANCED: Print phone statistics
        if enrich_with_email and self.apollo_enricher:
            verified_emails = sum(1 for p in data if p.get('verified_email') and p.get('email_verification_status') == 'verified')
            guessed_emails = sum(1 for p in data if p.get('verified_email') and p.get('email_verification_status') == 'guessed')
            likely_emails = sum(1 for p in data if p.get('verified_email') and p.get('email_verification_status') == 'likely')
            extrapolated_emails = sum(1 for p in data if p.get('verified_email') and p.get('email_verification_status') == 'extrapolated')
            company_pages = sum(1 for p in data if p.get('email_verification_status') == 'company_page_not_supported')
            unavailable = sum(1 for p in data if p.get('email_verification_status') == 'unavailable')
            not_in_db = sum(1 for p in data if p.get('email_verification_status') == 'not_in_apollo_db')
            no_emails = sum(1 for p in data if not p.get('verified_email'))
            total_with_emails = verified_emails + guessed_emails + likely_emails + extrapolated_emails
            
            # ✅ NEW: Phone statistics
            phones_found = sum(1 for p in data if p.get('apollo_phone') and p.get('apollo_phone').strip())

            print(f"\n📊 EMAIL ENRICHMENT BREAKDOWN:")
            print(f"  ✓ Verified emails: {verified_emails}")
            print(f"  ~ Guessed emails: {guessed_emails}")
            print(f"  ~ Likely emails: {likely_emails}")
            print(f"  ≈ Extrapolated emails: {extrapolated_emails}")
            print(f"  ⊗ Company pages (skipped): {company_pages}")
            print(f"  ⚠️ Unavailable in Apollo: {unavailable}")
            print(f"  ⚠️ Not in Apollo DB: {not_in_db}")
            print(f"  ✗ No email found: {no_emails - company_pages - unavailable - not_in_db}")

            if len(data) > 0:
                enrichment_rate = (total_with_emails / len(data) * 100)
                verified_rate = (verified_emails / len(data) * 100) if verified_emails > 0 else 0
                phone_rate = (phones_found / len(data) * 100)
                print(f"  📈 Total enrichment rate: {enrichment_rate:.1f}%")
                print(f"  📈 Verified email rate: {verified_rate:.1f}%")
                print(f"  📞 Phone number rate: {phone_rate:.1f}%")  # ✅ NEW: Phone statistics

        # Print company analysis statistics if enabled
        if analyze_companies and self.company_analyzer:
            industries_found = sum(1 for p in data if p.get('company_industry') and p.get('company_industry') != 'Data unavailable')
            categories_found = sum(1 for p in data if p.get('company_category') and p.get('company_category') != 'Data unavailable')
            
            print(f"\n📊 COMPANY ANALYSIS BREAKDOWN:")
            print(f"  ✓ Industries identified: {industries_found}/{len(data)} ({industries_found/len(data)*100:.1f}%)")
            print(f"  ✓ Categories identified: {categories_found}/{len(data)} ({categories_found/len(data)*100:.1f}%)")

        print(f"{'='*70}\n")

        return filename

    def close(self):
        """Close the browser"""
        if self.driver:
            self.driver.quit()
            print("Browser closed.")


# ============= USAGE EXAMPLES =============
if __name__ == "__main__":
    print("=" * 60)
    print("LinkedIn Lead Generation Scraper with Apollo + Mistral AI")
    print("✅ ENHANCED: Email + Phone enrichment + Post classification + Company Analysis")
    print("=" * 60)

    LINKEDIN_EMAIL = input("\nEnter your LinkedIn email: ").strip()
    LINKEDIN_PASSWORD = input("Enter your LinkedIn password: ").strip()

    if not LINKEDIN_EMAIL or not LINKEDIN_PASSWORD:
        print("Error: Email and password are required!")
        exit()

    # Optional: Apollo API key for email enrichment
    use_apollo = input("\nDo you want to enrich profiles with Apollo.io (emails + phones)? (y/n): ").strip().lower()
    if use_apollo == 'y':
        apollo_key = input("Enter your Apollo.io API key: ").strip()
        if not apollo_key:
            print("Warning: No Apollo API key provided. Email enrichment will be skipped.")
            apollo_key = None
    else:
        apollo_key = None

    # Optional: Mistral API key for post filtering AND company analysis
    use_mistral = input("\nDo you want to use Mistral AI features? (y/n): ").strip().lower()
    filter_posts = False
    analyze_companies = False
    mistral_key = None
    
    if use_mistral == 'y':
        mistral_key = input("Enter your Mistral API key: ").strip()
        if not mistral_key:
            print("Warning: No Mistral API key provided. Mistral features will be skipped.")
        else:
            filter_posts = input("  - Filter posts (remove individual hiring posts)? (y/n): ").strip().lower() == 'y'
            analyze_companies = input("  - Analyze companies (extract industry/category)? (y/n): ").strip().lower() == 'y'

    # Configure scraping
    print("\n" + "="*60)
    print("CONFIGURE SCRAPING")
    print("="*60)

    posts_config = input("\nHow many posts per URL? (press Enter for default 10): ").strip()
    if posts_config:
        try:
            posts_per_url = int(posts_config)
        except ValueError:
            print("Invalid number, using default (10)")
            posts_per_url = None
    else:
        posts_per_url = None

    # Define your search URLs
    search_urls = [
        "https://www.linkedin.com/search/results/content/?keywords=looking%20for%20marketing%20agency&origin=GLOBAL_SEARCH_HEADER&sid=hyx",
        "https://www.linkedin.com/search/results/content/?keywords=seeking%20marketing%20agency&origin=GLOBAL_SEARCH_HEADER&sid=(5z",
        "https://www.linkedin.com/search/results/content/?keywords=hiring%20marketing%20agency%20partners&origin=GLOBAL_SEARCH_HEADER&sid=Ath",
    ]

    print(f"\n{'='*60}")
    print(f"URLs to scrape: {len(search_urls)}")
    if posts_per_url:
        print(f"Posts per URL: {posts_per_url}")
        print(f"Estimated total posts: {len(search_urls) * posts_per_url}")
    else:
        print(f"Posts per URL: 10 (default)")
        print(f"Estimated total posts: {len(search_urls) * 10}")
    print(f"{'='*60}\n")

    print("Starting scraper...\n")

    # Initialize scraper
    scraper = LinkedInLeadScraper(LINKEDIN_EMAIL, LINKEDIN_PASSWORD, apollo_key, mistral_key)

    try:
        # Login to LinkedIn
        scraper.login()

        # Scrape multiple URLs
        all_leads = scraper.scrape_multiple_urls(search_urls, posts_per_url=posts_per_url)

        # Save to CSV with optional features
        scraper.save_to_csv(
            all_leads, 
            enrich_with_email=(apollo_key is not None),
            filter_with_mistral=filter_posts,
            analyze_companies=analyze_companies
        )

    except Exception as e:
        print(f"Error occurred: {e}")

    finally:
        # Close browser
        scraper.close()