#this is weekly.py
import sys
import io
import asyncio
import platform
import time
from main import call_llm
import os
import socket
# ==================== GLOBAL SELENIUM DRIVER ====================
_global_driver = None

def get_shared_driver():
    """Get or create a single shared Selenium driver"""
    global _global_driver
    if _global_driver is None:
        _global_driver = get_shared_driver()  
    return _global_driver

def cleanup_shared_driver():
    """Close the shared driver"""
    global _global_driver
    if _global_driver:
        try:
            _global_driver.quit()
            print("✓ Browser closed")
        except:
            pass
        _global_driver = None
# Add this with other imports
from inc42 import scrape_inc42_funding_table, search_duckduckgo_selenium as inc42_search, clean_text
LLM_PROVIDER = "mistral"
# ✅ Configure network settings
os.environ['PYTHONHTTPSVERIFY'] = '0'  # Disable SSL verification if behind proxy
socket.setdefaulttimeout(60)  # Increase global timeout to 60 seconds

# If behind corporate proxy, add:
# os.environ['HTTP_PROXY'] = 'http://proxy.company.com:8080'
# os.environ['HTTPS_PROXY'] = 'http://proxy.company.com:8080'
# LLM configurations (imported from main.py logic)
LLM_CONFIGS = {
    "Mistral": {"url": "https://api.mistral.ai/v1/chat/completions", "model": "mistral-small-latest", "auth_header": "Bearer"},
    "Claude": {"url": "https://api.anthropic.com/v1/messages", "model": "claude-3-5-sonnet-20241022", "auth_header": "x-api-key", "anthropic_version": "2023-06-01"},
    "Openai": {"url": "https://api.openai.com/v1/chat/completions", "model": "gpt-4o", "auth_header": "Bearer"},
    "Gemini": {"url": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions", "model": "gemini-2.0-flash", "auth_header": "Bearer"},
    "Deepseek": {"url": "https://api.deepseek.com/chat/completions", "model": "deepseek-chat", "auth_header": "Bearer"},
    "Qwen": {"url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions","model": "qwen-plus","auth_header": "Bearer"},
    "Perplexity": {"url": "https://api.perplexity.ai/chat/completions","model": "sonar","auth_header": "Bearer"},
    "Llama": {"url": "https://api.llama-api.com/chat/completions","model": "llama-3.3-70b","auth_header": "Bearer"}
}
# Safe encoding setup that works with StringIO and normal stdout
try:
    # Only reconfigure if it's actual stdout, not StringIO
    if hasattr(sys.stdout, "reconfigure") and not isinstance(sys.stdout, io.StringIO):
        sys.stdout.reconfigure(encoding='utf-8')
except:
    pass

# Set default encoding
import locale
try:
    locale.setlocale(locale.LC_ALL, 'en_IN.UTF-8')
except:
    try:
        locale.setlocale(locale.LC_ALL, 'C.UTF-8')
    except:
        pass

# ==================== CRITICAL: Windows Fix MUST be FIRST ====================
if platform.system() == 'Windows':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    print("✓ Windows event loop policy set for Playwright")

from datetime import datetime
this_month = datetime.now().strftime("%B")

# Continue with your other imports...


import requests
from time import sleep
import random
import re
import os
import pandas as pd
from bs4 import BeautifulSoup
from typing import Optional, List
from tqdm import tqdm
import json

# Selenium imports
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.keys import Keys
import zipfile
import subprocess
import shutil

# ==================== CONFIG ====================
FOUNDERSDAY_DIRECT_URL = "https://www.foundersday.co/startup-funding-tracker"
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"
MAX_PAGES_FOUNDERSDAY = 2

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)
ACCEPT_LANGUAGE = "en-US,en;q=0.9"
MAX_RETRIES = 3
selenium_driver = None  # Global Selenium driver instance

# Global variables for API key management
current_mistral_key = None
mistral_key_1 = None
mistral_key_2 = None
current_key_index = 0

# Global variables for API key management
current_mistral_key = None
mistral_key_1 = None
mistral_key_2 = None
current_key_index = 0

def initialize_api_keys(key1, key2):
    """Initialize the API keys for rotation"""
    global api_key_1, api_key_2, current_api_key, current_key_index  # ✅ Updated
    
    if not key1:
        print("❌ ERROR: Primary API key (key1) is empty!")
        print(f"   key1 value: '{key1}'")
        print(f"   key2 value: '{key2}'")
        raise ValueError("Primary API key cannot be empty!")
    
    api_key_1 = key1           # ✅ Updated
    api_key_2 = key2 if key2 else key1
    current_api_key = key1     # ✅ Updated
    current_key_index = 0
    
    print(f"✓ API keys initialized:")
    print(f"  • Primary key: {key1[:10]}...{key1[-4:]}")
    print(f"  • Secondary key: {key2[:10] if key2 else 'Using primary'}...{key2[-4:] if key2 else ''}")
    print(f"  • Starting with key 1")

def rotate_api_key():
    """Rotate to the next API key"""
    global current_api_key, current_key_index  # ✅ Updated
    current_key_index = 1 - current_key_index
    current_api_key = api_key_1 if current_key_index == 0 else api_key_2  # ✅ Updated
    print(f"🔄 Rotated to API key {current_key_index + 1}")
    return current_api_key  # ✅ Updated

# ==================== SELENIUM SETUP ====================
def get_chrome_version():
    try:
        if platform.system() == "Windows":
            result = subprocess.run(
                ['reg', 'query', 'HKEY_CURRENT_USER\Software\Google\Chrome\BLBeacon', '/v', 'version'],
                capture_output=True, text=True
            )
            return result.stdout.split()[-1]
        elif platform.system() == "Darwin":
            result = subprocess.run(
                ['/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', '--version'],
                capture_output=True, text=True
            )
            return result.stdout.split()[-1].split()[0]
        else:
            result = subprocess.run(['google-chrome', '--version'], capture_output=True, text=True)
            return result.stdout.split()[-1]
    except:
        return None


def setup_selenium_driver():
    

    # Random user agents to rotate
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    ]
    
    selected_ua = random.choice(user_agents)
    
    opts = webdriver.ChromeOptions()
    opts.add_argument("--headless")
    opts.add_argument(f'--user-agent={selected_ua}')
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--start-maximized")
    
    # Add random window size
    window_sizes = ['1920,1080', '1366,768', '1536,864', '1440,900']
    opts.add_argument(f'--window-size={random.choice(window_sizes)}')
    
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option('useAutomationExtension', False)
    
    # Add preferences
    prefs = {
        "credentials_enable_service": False,
        "profile.password_manager_enabled": False,
        "profile.default_content_setting_values.notifications": 2
    }
    opts.add_experimental_option("prefs", prefs)

    if os.path.exists("/usr/bin/chromium"):  # Streamlit Cloud
      opts.binary_location = "/usr/bin/chromium"
      service = Service("/usr/bin/chromedriver")
    else:  # Local
      from webdriver_manager.chrome import ChromeDriverManager
      service = Service(ChromeDriverManager().install())

    driver = webdriver.Chrome(service=service, options=opts)
    
    # Execute stealth scripts
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    
    print(f"✓ Browser started with stealth mode (User-Agent: {selected_ua[:50]}...)")
    
    # Test DuckDuckGo connection
    driver.get("https://duckduckgo.com")
    time.sleep(2)
    print("✓ Browser ready for research (using DuckDuckGo)")
    
    return driver

def get_selenium_driver():
    global selenium_driver  # ✅ ENSURE THIS LINE EXISTS
    if selenium_driver is None:
        selenium_driver = get_shared_driver()  
    return selenium_driver

def cleanup_selenium_driver():
    global selenium_driver  # ✅ ADD THIS LINE
    if selenium_driver:
        try:
            selenium_driver.quit()
        except:
            pass
        selenium_driver = None

# ==================== MISTRAL AI FUNCTIONS WITH KEY ROTATION ====================
def convert_to_inr(amount_str):
    """Converts USD/other currency amounts to INR approximation."""
    if not amount_str:
        return ""

    usd_rate = 83

    match = re.search(r'\$\s*([\d.]+)\s*([MBK])', amount_str, re.IGNORECASE)
    if match:
        value = float(match.group(1))
        unit = match.group(2).upper()

        multipliers = {'K': 1000, 'M': 1000000, 'B': 1000000000}
        numeric_value = value * multipliers.get(unit, 1)
        inr_value = numeric_value * usd_rate

        if inr_value >= 10000000:
          return f"INR {inr_value / 10000000:.2f} Cr"
        elif inr_value >= 100000:
          return f"INR {inr_value / 100000:.2f} Lakh"
        else:
          return f"INR {inr_value:,.0f}"

    # Check if already has rupee symbol or INR
    if 'INR' in amount_str or '₹' in amount_str or '\u20B9' in amount_str:
       # Replace any garbled symbols with INR
       amount_str = re.sub(r'[^\x00-\x7F]+', 'INR ', amount_str)
       return amount_str

    return amount_str

def is_valid_founder_name(name):
    """Validates if extracted text is actually a founder name."""
    if not name or len(name) < 2:
        return False

    name = name.strip()

    # Check for dates
    if re.match(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', name):
        return False

    # Too many numbers
    if len(re.findall(r'\d', name)) / len(name) > 0.3:
        return False

    # All caps short abbreviation
    if re.match(r'^[A-Z]{1,4}$', name):
        return False

    # Single word short names (likely invalid)
    if ' ' not in name and len(name) < 3:
        return False

    # Must contain letters
    if not re.search(r'[a-zA-Z]', name):
        return False

    # Check for keywords
    keywords = ['saas', 'b2b', 'b2c', 'fintech', 'healthtech', 'platform', 'startup', 
                'investors', 'founder', 'founders', 'fundraise', 'india', 'company',
                'confidence', 'sharma confidence', 'ceo', 'cto', 'cfo']
    if any(kw in name.lower() for kw in keywords):
        return False
    
    # Reject single-word names (likely incomplete)
    if ' ' not in name:
        return False
    
    # Reject names that are just "Lastname\nWord" pattern or have line breaks
    if '\n' in name:
        return False
    
    # Check for valid name structure (at least firstname + lastname)
    parts = name.split()
    if len(parts) < 2:
        return False
    
    # Each part should be mostly alphabetic
    for part in parts:
        if not part.isalpha():
            return False

    return True

def validate_founder_against_company(company_name, founder_name):
    """
    Check if founder name is similar to company name and should be removed.

    Args:
        company_name (str): The company name
        founder_name (str): The founder name to validate

    Returns:
        bool: True if founder name is valid, False if it should be removed
    """
    if not company_name or not founder_name:
        return True

    # Normalize names for comparison
    company_clean = re.sub(r'[^a-zA-Z0-9\s]', '', company_name.lower().strip())
    founder_clean = re.sub(r'[^a-zA-Z0-9\s]', '', founder_name.lower().strip())

    # Remove common company suffixes for better comparison
    company_suffixes = ['inc', 'ltd', 'llc', 'corp', 'corporation', 'company', 'co', 'pvt', 'private', 'limited']
    company_words = company_clean.split()
    company_words = [word for word in company_words if word not in company_suffixes]
    company_clean = ' '.join(company_words)

    # Direct match
    if company_clean == founder_clean:
        print(f"  🚫 Removing founder '{founder_name}' - matches company name '{company_name}'")
        return False

    # Check if founder name is contained in company name or vice versa
    if len(founder_clean) > 3:  # Only check for meaningful names
        if founder_clean in company_clean or company_clean in founder_clean:
            print(f"  🚫 Removing founder '{founder_name}' - too similar to company name '{company_name}'")
            return False

    # Check similarity ratio (if names share 80%+ words)
    company_word_set = set(company_words)
    founder_word_set = set(founder_clean.split())

    if len(founder_word_set) > 0:
        similarity = len(company_word_set.intersection(founder_word_set)) / len(founder_word_set)
        if similarity >= 0.8:
            print(f"  🚫 Removing founder '{founder_name}' - {similarity*100:.0f}% similarity to company name '{company_name}'")
            return False

    return True

def filter_category_field(category):
    """Filter category field to only allow B2B/B2C and remove unwanted categories."""
    if not category:
        return ""

    category = category.strip()

    # Split by common separators
    category_parts = []
    for part in category.replace(',', '|').replace('/', '|').replace('&', '|').replace(' and ', '|').split('|'):
        category_parts.extend(part.strip().split())

    # Clean and filter parts
    allowed_terms = []
    for part in category_parts:
        part_lower = part.lower().strip()

        # Only keep B2B/B2C related terms
        if part_lower in ['b2b', 'b2c', 'b2g']:
            allowed_terms.append(part_lower.upper())
        elif part_lower in ['business', 'consumer', 'government']:
            # Convert to B2B/B2C format
            if part_lower == 'business':
                allowed_terms.append('B2B')
            elif part_lower == 'consumer':
                allowed_terms.append('B2C')
            elif part_lower == 'government':
                allowed_terms.append('B2G')

    # Remove duplicates while preserving order
    unique_terms = []
    for term in allowed_terms:
        if term not in unique_terms:
            unique_terms.append(term)

    return ', '.join(unique_terms) if unique_terms else ""
def is_valid_company_name(name):
    """Validate if extracted text is a valid company name, including domain-like names."""
    if not name or len(name) < 2:
        return False
    
    name = name.strip()
    
    # Allow domain-like names (.ai, .dev, .io, .com, etc.)
    domain_pattern = r'^[A-Za-z0-9]+\.(ai|dev|io|com|co|app|tech)$'
    if re.match(domain_pattern, name, re.IGNORECASE):
        return True
    
    # Check if it's a sentence (multiple words with common sentence structure)
    if len(name.split()) > 6:
        return False
    
    # Check for sentence indicators
    sentence_indicators = ['. ', '? ', '! ', ' is ', ' are ', ' was ', ' were ', ' the ']
    if any(indicator in name.lower() for indicator in sentence_indicators):
        return False
    
    # Must contain letters
    if not re.search(r'[a-zA-Z]', name):
        return False
    
    return True
def extract_with_mistral(card_text, llm_name, retry=0, max_retries=3):
    """Uses LLM REST API with key rotation to intelligently extract company data from card text."""
    global current_mistral_key, current_api_key

    try:
        prompt = f"""You are an expert at extracting startup funding data. Extract ONLY valid JSON from this text.

TEXT:
{card_text}

INSTRUCTIONS:
- Company name: Extract ONLY the actual company/startup name. This is usually:
  * The FIRST line of text in the card
  * The SHORTEST text (typically 1-4 words)
  * In BOLD or larger font
  * May include domain extensions like .ai, .dev, .io (e.g., "Reo.dev", "Acme.ai", "Matters.AI")
  * NOT a description or sentence
  * Examples of valid names: "Acme Corp", "TechStart", "HealthPlus AI", "Reo.dev", "Matters.AI"
  * INVALID: Long sentences, descriptions, or phrases like "A platform that..."
- Funding amount: extract the amount with proper currency symbol (e.g., $4M, ₹50Cr, INR 30Cr)
  * If you see ₹ symbol, write it as "INR" instead (e.g., "INR 50 Cr" not "₹50 Cr")
- Funding round: Seed, Pre-seed, Series A, Series B, etc.
- Founders: Extract ONLY actual founder names (first and last name). DO NOT include:
  * Dates like "10/8/2025", "October 2025", etc
  * Single word names that are company names
  * Numbers or special characters
  * Company keywords (SAAS, B2B, B2C, AI, etc)
  * Investor names or VC firms
  First valid founder goes to founder_name, rest to important_person
- Industry/sectors: tags under company name (SAAS, AI, healthtech, cleantech, etc)
- Category: B2B/B2C/B2G or domain (healthtech, cleantech, fintech, etc)
- Description: 1-2 sentence description of what the company does (NOT the company name)
- Date: Extract any date mentioned (format: 'MMM DD, YYYY' or 'MMM DD-DD, YYYY')

VALID COMPANY NAME EXAMPLES:
- "Reo.dev" ✓ (domain-like name)
- "Matters.AI" ✓ (domain-like name with capital letters)
- "Acme.ai" ✓ (domain-like name)
- "TechCorp" ✓ (short brand name)
- "HealthPlus AI" ✓ (brand with tech suffix)

INVALID COMPANY NAME EXAMPLES:
- "A platform that connects doctors with patients" ✗ (description)
- "The company raised $4M" ✗ (sentence)
- "Investors" ✗ (not a company name)
- "Founded in 2020" ✗ (sentence fragment)

CRITICAL:
- Company name should be SHORT (1-6 words max), NOT a full sentence
- Company name CAN include .ai, .dev, .io, .tech domains - these are VALID
- Description should be a sentence explaining what the company does
- NEVER put description text in company_name field
- NEVER put "Investors" or similar generic terms as company name

Return ONLY valid JSON (no markdown, no code blocks):
{{
  "company_name": "Short company name (can include domains like Reo.dev, Matters.AI)",
  "funding_amount": "Amount with INR/$ (e.g., INR 4 Cr or $4M)",
  "funding_round": "Seed/Series A etc",
  "founder_name": "First founder name only",
  "important_person": "Other founders comma separated",
  "industry": "comma separated (SAAS, AI, FinTech)",
  "category": "B2B, B2C, or domain",
  "description": "What company does in 1-2 sentences (not the name)",
  "date": "Extract date in format 'MMM DD, YYYY' or 'MMM DD-DD, YYYY' (e.g., 'Oct 20, 2025' or 'Sept 20-26, 2025')"
}}"""

        # Use universal LLM caller
        response_text = call_llm(llm_name, current_api_key, prompt, max_tokens=500)

        if not response_text:
            print(f"⚠️ API error with key {current_key_index + 1}")
            if retry < max_retries:
                rotate_api_key()
                wait_time = 2 ** retry
                print(f"🔄 Retrying with key {current_key_index + 1} in {wait_time}s...")
                time.sleep(wait_time)
                return extract_with_mistral(card_text, llm_name, retry + 1, max_retries)
            else:
                print(f"⚠️ API error after {max_retries} retries with both keys")
                return None

        # Clean response
        response_text = re.sub(r'```json\n?|\n?```', '', response_text)
        response_text = re.sub(r'```\n?', '', response_text)
        response_text = response_text.strip()

        try:
            data = json.loads(response_text)
        except json.JSONDecodeError:
            if retry < max_retries:
                rotate_api_key()
                wait_time = 2 ** retry
                print(f"⚠️ JSON parsing error. Retrying with key {current_key_index + 1} in {wait_time}s...")
                time.sleep(wait_time)
                return extract_with_mistral(card_text, llm_name, retry + 1, max_retries)
            else:
                print(f"⚠️ JSON parsing failed after {max_retries} retries")
                return None
        
        # ✅ IMPROVED POST-PROCESSING VALIDATION
        company_name = data.get("company_name", "").strip()
        
        # Check for invalid generic terms
        invalid_terms = ['investors', 'investment', 'funding', 'venture', 'capital', 'founded', 'raised']
        if company_name.lower() in invalid_terms:
            print(f"  ⚠️ Invalid generic term detected: '{company_name}'")
            # Try to extract first line from card_text as company name
            first_line = card_text.split('\n')[0].strip()
            if is_valid_company_name(first_line) and first_line.lower() not in invalid_terms:
                data["company_name"] = first_line
                print(f"  ✓ Fixed company name to: '{first_line}'")
            else:
                print(f"  ⚠️ Could not fix company name, skipping validation")
                return None
        
        # Validate company name - but SKIP validation for domain-like names
        domain_pattern = r'^[A-Za-z0-9]+\.(ai|dev|io|com|co|app|tech)$'
        is_domain_name = bool(re.match(domain_pattern, company_name, re.IGNORECASE))
        
        if is_domain_name:
            # Domain names are ALWAYS valid, skip further validation
            print(f"  ✅ Valid domain name detected: '{company_name}'")
        elif not is_valid_company_name(company_name):
            print(f"  ⚠️ Invalid company name detected: '{company_name[:50]}...'")
            # Try to extract first line from card_text as company name
            first_line = card_text.split('\n')[0].strip()
            if is_valid_company_name(first_line):
                data["company_name"] = first_line
                print(f"  ✓ Fixed company name to: '{first_line}'")
            else:
                print(f"  ⚠️ Could not fix company name")
        
        # If company name looks like a description (too long), try to fix
        # BUT only if it's NOT a domain name
        if not is_domain_name and (len(company_name) > 60 or company_name.count('.') > 2):
            print(f"  ⚠️ Company name looks like description: '{company_name[:50]}...'")
            # Try to extract first line from card_text as company name
            first_line = card_text.split('\n')[0].strip()
            if len(first_line) < 60 and len(first_line) > 0:
                data["company_name"] = first_line
                print(f"  ✓ Fixed company name to: '{first_line}'")
        
        # Clean funding amount - replace garbled rupee symbols
        funding_amount = data.get("funding_amount", "")
        if funding_amount:
            # Replace any non-ASCII characters with INR
            funding_amount = re.sub(r'[^\x00-\x7F]+', 'INR ', funding_amount)
            # Clean up double spaces
            funding_amount = re.sub(r'\s+', ' ', funding_amount).strip()
            data["funding_amount"] = funding_amount
        
        return data

    except Exception as e:
        if retry < max_retries:
            # Try rotating key on any error
            rotate_api_key()
            wait_time = 2 ** retry
            print(f"⚠️ Error: {e}. Retrying with key {current_key_index + 1} in {wait_time}s...")
            time.sleep(wait_time)
            return extract_with_mistral(card_text, llm_name, retry + 1, max_retries)
        else:
            print(f"⚠️ Failed after {max_retries} retries with both keys: {e}")
            return None
        
def extract_fallback(card_text):
    """Fallback extraction using regex when Mistral fails."""
    data = {
        "company_name": "",
        "funding_amount": "",
        "funding_round": "",
        "founder_name": "",
        "important_person": "",
        "industry": "",
        "category": "",
        "description": ""
    }

    lines = card_text.split('\n')

    for line in lines:
        if line.strip() and len(line.strip()) < 100:
            data["company_name"] = line.strip()
            break

    funding_match = re.search(r'\$\s*([\d.]+\s*[MBK])', card_text, re.IGNORECASE)
    if funding_match:
        data["funding_amount"] = f"${funding_match.group(1)}"

    round_match = re.search(r'(Seed|Pre-seed|Series [A-Z])', card_text, re.IGNORECASE)
    if round_match:
        data["funding_round"] = round_match.group(1)

    founders_section = re.search(r'Founder[s]*\s*\n(.*?)(?:\n[A-Z]|$)', card_text, re.DOTALL | re.IGNORECASE)
    if founders_section:
        founders_text = founders_section.group(1)
        founder_links = re.findall(r'([A-Z][a-z]+ [A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)', founders_text)
        valid_founders = [f for f in founder_links if is_valid_founder_name(f)]

        if valid_founders:
            data["founder_name"] = valid_founders[0]
            if len(valid_founders) > 1:
                data["important_person"] = ", ".join(valid_founders[1:])

    desc_lines = [l.strip() for l in lines if l.strip() and len(l.strip()) > 20]
    if desc_lines:
        data["description"] = " ".join(desc_lines[:2])[:150]

    return data

def is_noise_text(text):
    """Checks if text is noise/newsletter signup text."""
    noise_keywords = [
        "receive a weekly summary",
        "unsubscribe anytime",
        "filter by sector",
        "newsletter",
        "subscribe",
        "email",
        "sign up",
        "get updates"
    ]
    return any(keyword in text.lower() for keyword in noise_keywords)

# ==================== FOUNDERSDAY SCRAPER (MISTRAL AI) ====================
def scrape_foundersday_with_mistral(output_csv, llm_name):
    """Main FoundersDay scraper using Mistral AI - creates initial CSV."""
    driver = get_selenium_driver()
    all_data = []
    seen_companies = set()  # Track companies to detect duplicate pages

    try:
        print("\n" + "=" * 70)
        print("🚀 PHASE 1: FoundersDay Scraper (Mistral AI)")
        print(f"📄 Scraping up to {MAX_PAGES_FOUNDERSDAY} pages")
        print("=" * 70)

        for page in range(1, MAX_PAGES_FOUNDERSDAY + 1):
            print(f"\n{'=' * 70}")
            print(f"📖 Page {page}/{MAX_PAGES_FOUNDERSDAY}")
            print(f"{'=' * 70}")

            # Try different pagination URL formats
            if page == 1:
                url = FOUNDERSDAY_DIRECT_URL
            else:
                # FoundersDay uses ?page=X format
                url = f"{FOUNDERSDAY_DIRECT_URL}?74bb1511_page={page}"

            print(f"🔗 Loading: {url}")
            driver.get(url)
            time.sleep(1)

            # Verify we're on a new page by checking URL
            current_url = driver.current_url
            print(f"📍 Current URL: {current_url}")

            # Scroll to load content
            driver.execute_script("window.scrollTo(0, 1000);")
            time.sleep(1)
            driver.execute_script("window.scrollTo(0, 0);")
            time.sleep(1)

            # Additional scroll to ensure all content loads
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1)

            # Find cards
            selectors_to_try = [
                "//div[@role='listitem' and contains(@class, 'w-dyn-item')]",
                "//div[contains(@class, 'w-dyn-item')]",
                "//div[contains(@class, 'div-block-2')]",
                "//div[@role='listitem']"
            ]

            cards = []
            for selector in selectors_to_try:
                try:
                    found = driver.find_elements(By.XPATH, selector)
                    if len(found) >= 1:
                        cards = found
                        print(f"✅ Found {len(cards)} cards using: {selector}")
                        break
                except:
                    continue

            if not cards:
                print("⚠️ No company cards found")
                if page > 1:
                    print("🛑 Stopping pagination - likely reached end of available pages")
                    break
                continue

            print(f"📊 Processing {len(cards)} cards with Mistral AI...")

            # Track companies on this page to detect duplicates
            page_companies = []

            for idx, card in enumerate(cards, 1):
                try:
                    card_text = card.text.strip() if card.text else ""

                    if not card_text or len(card_text) < 30 or is_noise_text(card_text):
                        continue

                    print(f"\n🤖 Processing card #{idx}...")

                    extracted = extract_with_mistral(card_text, llm_name)

                    if not extracted or not extracted.get("company_name"):
                        print(f"⚠️ Mistral failed. Using fallback...")
                        extracted = extract_fallback(card_text)
                        if not extracted or not extracted.get("company_name"):
                            continue

                    company_name = extracted.get("company_name", "").strip()

                    # Check if this company was already seen (duplicate detection)
                    company_name_lower = company_name.lower()
                    if company_name_lower in seen_companies:
                        print(f"⚠️ Duplicate detected: {company_name} (already scraped)")
                        continue

                    funding_amount = extracted.get("funding_amount", "")
                    inr_amount = convert_to_inr(funding_amount) if funding_amount else ""

                    founder_name = extracted.get("founder_name", "").strip()
                    important_person = extracted.get("important_person", "").strip()

                    # Validate founder name (existing validation)
                    if founder_name and not is_valid_founder_name(founder_name):
                        founder_name = ""

                    # NEW: Validate founder name against company name
                    if founder_name and not validate_founder_against_company(company_name, founder_name):
                        founder_name = ""

                    # Process important_person field
                    if important_person:
                        persons = [p.strip() for p in important_person.split(",")]
                        valid_persons = []
                        for person in persons:
                            if is_valid_founder_name(person) and validate_founder_against_company(company_name, person):
                                valid_persons.append(person)
                        important_person = ", ".join(valid_persons)

                    data_row = {
                        "Company_Name": company_name,
                        "Funding_Amount": inr_amount,
                        "Country": "India",
                        "Founder_Name": founder_name,
                        "Important_Person": important_person,
                        "Email": "",
                        "LinkedIn_URL": "",
                        "Industry_Sector": extracted.get("industry", "").strip(),
                        "Category": filter_category_field(extracted.get("category", "")),
                        "Description": extracted.get("description", "").strip(),
                        "Date": extracted.get("date", "").strip()
                    }

                    all_data.append(data_row)
                    seen_companies.add(company_name_lower)
                    page_companies.append(company_name)

                    print(f"✅ Extracted: {data_row['Company_Name']}")
                    if data_row["Funding_Amount"]:
                        print(f"   Funding: {data_row['Funding_Amount']}")
                    if data_row["Founder_Name"]:
                        print(f"   Founder: {data_row['Founder_Name']}")

                except Exception as e:
                    print(f"⚠️ Error processing card #{idx}: {e}")
                    continue

            # Check if we got any new companies on this page
            if not page_companies:
                print(f"\n⚠️ No new companies found on page {page}")
                if page > 1:
                    print("🛑 Stopping pagination - likely on same page or end reached")
                    break
            else:
                print(f"\n✅ Page {page}: Extracted {len(page_companies)} new companies")
                print(f"   First few: {', '.join(page_companies[:3])}")

            # Wait before next page
            if page < MAX_PAGES_FOUNDERSDAY:
                print(f"\n⏳ Waiting 5 seconds before loading page {page + 1}...")
                time.sleep(2)

        # Create initial CSV with FoundersDay data
        if all_data:
            df = pd.DataFrame(all_data)
            df.drop_duplicates(subset=["Company_Name"], keep="first", inplace=True)
            df.to_csv(output_csv, index=False, encoding='utf-8-sig')
            print(f"\n✅ FoundersDay Phase Complete: {len(df)} companies saved to {output_csv}")
            return len(df)
        else:
            print("\n⚠️ No FoundersDay data extracted")
            # Create empty CSV with headers
            df = pd.DataFrame(columns=[
    "Company_Name", "Funding_Amount", "Country", "Founder_Name",
    "Important_Person", "Email", "LinkedIn_URL", "Description","Industry_Sector", "Category", "Date"
])
            df.to_csv(output_csv, index=False)
            return 0

    except Exception as e:
        print(f"\n❌ Error in FoundersDay scraping: {e}")
        import traceback
        traceback.print_exc()
        cleanup_selenium_driver()
        return 0

def search_duckduckgo_selenium(query, max_retries=3):
    """Use Selenium to search DuckDuckGo and return the first relevant link."""
    for attempt in range(max_retries):
        try:
            driver = get_selenium_driver()
            print(f"  🔍 Searching DuckDuckGo: {query}")

            # Navigate to DuckDuckGo
            driver.get("https://duckduckgo.com/")
            time.sleep(random.uniform(1, 2))

            # Find search box and enter query
            search_box = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.NAME, "q"))
            )
            search_box.clear()
            search_box.send_keys(query + Keys.RETURN)
            
            # Random human-like delay
            delay = random.uniform(1, 2)
            print(f"  ⏳ Waiting {delay:.1f}s (human-like behavior)...")
            time.sleep(delay)

            # Wait for results to load
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "article[data-testid='result']"))
            )

            # Find result links - DuckDuckGo uses different selectors
            result_links = driver.find_elements(By.CSS_SELECTOR, "article[data-testid='result'] a[href]")

            # Prioritize specific domains
            for link in result_links:
                try:
                    href = link.get_attribute("href")
                    if not href:
                        continue

                    # Skip DuckDuckGo's own pages
                    if any(x in href for x in ["duckduckgo.com", "youtube.com", "webcache"]):
                        continue

                    # Prefer these domains first
                    if any(domain in href for domain in ["yourstory.com", "entrackr.com", "startuptalky.com", "crunchbase.com", "alleywatch.com", "techcrunch.com"]):
                        print(f"  ✓ Found: {href}")
                        return href
                except:
                    continue

            # Fallback: return any valid link
            for link in result_links:
                try:
                    href = link.get_attribute("href")
                    if href and href.startswith("http") and "duckduckgo.com" not in href:
                        print(f"  ✓ Found (alternative): {href}")
                        return href
                except:
                    continue

            print("  ⚠️ No relevant link found")
            return None

        except Exception as e:
            print(f"  ⚠️ Search error (attempt {attempt + 1}/{max_retries}): {str(e)[:200]}")
            if attempt < max_retries - 1:
                sleep(random.uniform(4, 7))
            else:
                return None

    return None

def get_latest_article_link(directory_url, source_name):
    """
    Navigate to a directory page and extract the first (latest) article link.
    
    Args:
        directory_url: URL of the directory/listing page
        source_name: Name of the source ('YOURSTORY', 'ENTRACKR', or 'INC42')
    
    Returns:
        URL of the latest article or None
    """
    try:
        driver = get_selenium_driver()
        print(f"  📂 Opening directory page: {directory_url}")
        driver.get(directory_url)
        time.sleep(random.uniform(2,3))
        
        # Scroll to ensure content loads
        driver.execute_script("window.scrollTo(0, 500);")
        time.sleep(1)
        driver.execute_script("window.scrollTo(0, 0);")
        time.sleep(1)
        
        if source_name == 'YOURSTORY':
            # YourStory extraction code (existing)
            selectors = [
                "li.sc-c9f6afaa-0 a[href*='weekly-funding-roundup']",
                "a[href*='weekly-funding-roundup']",
                "div.sc-c9f6afaa-7 a[href^='/']",
                "li a[href*='funding']"
            ]
            
            for selector in selectors:
                try:
                    links = driver.find_elements(By.CSS_SELECTOR, selector)
                    if links:
                        first_link = links[0]
                        href = first_link.get_attribute("href")
                        
                        if href and href.startswith('/'):
                            href = f"https://yourstory.com{href}"
                        
                        if href and 'yourstory.com' in href:
                            print(f"  ✅ Found latest YourStory article: {href}")
                            return href
                except Exception as e:
                    print(f"  ⚠️ Selector {selector} failed: {e}")
                    continue
        
        elif source_name == 'ENTRACKR':
            # Entrackr extraction code (existing)
            try:
                html = driver.page_source
                soup = BeautifulSoup(html, "html.parser")
                
                og_url_tag = soup.find('meta', {'property': 'og:url'})
                if og_url_tag:
                    href = og_url_tag.get('content')
                    if href and 'entrackr.com' in href and 'funding-and-acquisitions' in href:
                        print(f"  ✅ Found latest Entrackr article from og:url: {href}")
                        return href
                
                print(f"  ⚠️ og:url method didn't find valid article link")
            except Exception as e:
                print(f"  ⚠️ og:url extraction failed: {e}")
            
            # Fallback methods for Entrackr...
            try:
                gallery_div = driver.find_element(By.CSS_SELECTOR, "div.gallery_content")
                if gallery_div:
                    first_link = gallery_div.find_element(By.CSS_SELECTOR, "a[href*='/report/weekly-funding-report/']")
                    if first_link:
                        href = first_link.get_attribute("href")
                        
                        if href and href.startswith('/'):
                            href = f"https://entrackr.com{href}"
                        
                        if href and 'entrackr.com' in href:
                            print(f"  ✅ Found latest Entrackr article (gallery): {href}")
                            return href
            except Exception as e:
                print(f"  ⚠️ Gallery_content method failed: {e}")
            
            selectors = [
                "div.gallery_content a[href*='funding-and-acquisitions']",
                "a[href*='weekly-funding-report'][href*='funding-and-acquisitions']",
                "div.feat-a-2 a[href*='/report/']",
                "a[href*='/report/weekly-funding-report/']"
            ]
            
            for selector in selectors:
                try:
                    links = driver.find_elements(By.CSS_SELECTOR, selector)
                    if links:
                        first_link = links[0]
                        href = first_link.get_attribute("href")
                        
                        if href and href.startswith('/'):
                            href = f"https://entrackr.com{href}"
                        
                        if href and 'entrackr.com' in href:
                            print(f"  ✅ Found latest Entrackr article (selector): {href}")
                            return href
                except Exception as e:
                    print(f"  ⚠️ Selector {selector} failed: {e}")
                    continue
        
        # ✅ ADD INC42 EXTRACTION HERE
        elif source_name == 'INC42':
            try:
                html = driver.page_source
                soup = BeautifulSoup(html, "html.parser")
                
                # Inc42 tag pages have article cards with data-card-id
                article_cards = soup.find_all('div', {'data-card-id': True})
                
                if article_cards:
                    print(f"  📰 Found {len(article_cards)} article cards on Inc42 tag page")
                    
                    for card in article_cards:
                        # Find link inside card
                        link = card.find('a', href=True)
                        if link:
                            article_url = link['href']
                            
                            # Make sure it's a full URL
                            if not article_url.startswith('http'):
                                article_url = 'https://inc42.com' + article_url
                            
                            # Check if it's a funding galore article
                            article_title = link.get('title', '').lower()
                            if any(keyword in article_url.lower() or keyword in article_title for keyword in ['funding', 'galore', 'raised', 'startups']):
                                print(f"  ✅ Found first Inc42 funding article: {article_url}")
                                return article_url
                    
                    # If no funding article found, return first article anyway
                    first_link = article_cards[0].find('a', href=True)
                    if first_link:
                        article_url = first_link['href']
                        if not article_url.startswith('http'):
                            article_url = 'https://inc42.com' + article_url
                        print(f"  ✅ Found first Inc42 article: {article_url}")
                        return article_url
                
                # Fallback: try different selectors
                article_links = soup.find_all('a', href=True)
                for link in article_links:
                    href = link['href']
                    if '/buzz/' in href or '/features/' in href:
                        if not href.startswith('http'):
                            href = 'https://inc42.com' + href
                        if 'funding' in href.lower() or 'galore' in href.lower():
                            print(f"  ✅ Found Inc42 article (fallback): {href}")
                            return href
                
                print("  ⚠️ Could not find any Inc42 article on tag page")
                return None
                
            except Exception as e:
                print(f"  ❌ Error extracting Inc42 article: {e}")
                import traceback
                traceback.print_exc()
                return None
        
        print(f"  ❌ Could not find latest article link for {source_name}")
        return None
        
    except Exception as e:
        print(f"  ❌ Error getting latest article link: {e}")
        import traceback
        traceback.print_exc()
        return None

def scrape_with_fallback(url):
    """Try multiple scraping methods with fallbacks."""
    print(f"  🌐 Attempting to scrape: {url}")

    try:
        print("  🔄 Method 1: Trying requests...")
        headers = {"User-Agent": USER_AGENT}
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        html = response.text
        if html and len(html) > 1000:
            print(f"  ✅ Requests scraping successful ({len(html)} chars)")
            return html
        else:
            print(f"  ⚠️ Requests returned insufficient content ({len(html)} chars)")
    except Exception as e:
        print(f"  ⚠️ Requests failed: {str(e)[:100]}")

    try:
        print("  🔄 Method 2: Falling back to Selenium...")
        driver = get_selenium_driver()
        print(f"  🌐 Selenium: Loading {url}")
        driver.get(url)
        
        # Verify we're on the correct page
        current_url = driver.current_url
        print(f"  📍 Selenium: Current URL after load: {current_url}")
        
        # Wait for content to load
        print("  ⏳ Waiting for page content to load...")
        time.sleep(1)
        
        # Scroll to load dynamic content
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1)
        driver.execute_script("window.scrollTo(0, 0);")
        time.sleep(1)
        
        html = driver.page_source
        if html and len(html) > 1000:
            print(f"  ✅ Selenium scraping successful ({len(html)} chars)")
            return html
        else:
            print(f"  ⚠️ Selenium returned insufficient content ({len(html)} chars)")
    except Exception as e:
        print(f"  ⚠️ Selenium failed: {str(e)[:100]}")
        import traceback
        traceback.print_exc()

    print("  ❌ All scraping methods failed")
    return None

def extract_country_from_text(text, company_name):
    """Extract country from article text."""
    india_keywords = ['india', 'indian', 'bangalore', 'mumbai', 'delhi', 'gurugram', 'hyderabad', 'chennai', 'pune']
    us_keywords = ['us', 'united states', 'american', 'new york', 'san francisco', 'california', 'silicon valley']
    uk_keywords = ['uk', 'united kingdom', 'london', 'british']
    singapore_keywords = ['singapore', 'singaporean']

    text_lower = text.lower()

    if any(keyword in text_lower for keyword in india_keywords):
        return "India"
    elif any(keyword in text_lower for keyword in us_keywords):
        return "United States"
    elif any(keyword in text_lower for keyword in uk_keywords):
        return "United Kingdom"
    elif any(keyword in text_lower for keyword in singapore_keywords):
        return "Singapore"

    return "Not specified"



def parse_yourstory(html, llm_name):
    """Parse YourStory using LLM AI for 'Key transactions' section."""
    soup = BeautifulSoup(html, "html.parser")
    companies = []

    # ✅ Extract date from page title tag (most reliable)
    article_date = ""
    try:
        # Method 1: Extract from <title> tag (BEST - as shown in your HTML)
        title_tag = soup.find('title')
        if title_tag:
            title_text = title_tag.get_text().strip()
            print(f"  📰 Found title tag: {title_text}")
            
            # Extract date pattern like "Sept 20-26" from title
            # Pattern: [Weekly funding roundup Sept 20-26] VC inflow...
            date_match = re.search(r'\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+(\d{1,2})\s*-\s*(\d{1,2})\b', title_text, re.IGNORECASE)
            
            if date_match:
                month = date_match.group(1)
                start_day = date_match.group(2)
                end_day = date_match.group(3)
                article_date = f"{month} {start_day}-{end_day}"
                print(f"  📅 Extracted date from title tag: {article_date}")
        
        # Method 2: Fallback to h1 headline if title tag fails
        if not article_date:
            title_selectors = ['h1', 'h1.entry-title', '.headline', '.article-title', 'h1.post-title', 'h1.entry-title.post-title']
            for selector in title_selectors:
                title_element = soup.select_one(selector)
                if title_element:
                    title_text = title_element.get_text().strip()
                    print(f"  📰 Found h1 headline: {title_text[:100]}...")
                    
                    date_match = re.search(r'\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+(\d{1,2})\s*-\s*(\d{1,2})\b', title_text, re.IGNORECASE)
                    
                    if date_match:
                        month = date_match.group(1)
                        start_day = date_match.group(2)
                        end_day = date_match.group(3)
                        article_date = f"{month} {start_day}-{end_day}"
                        print(f"  📅 Extracted date from h1: {article_date}")
                        break
        
        if not article_date:
            print(f"  ⚠️ Could not extract date from title or headline")
    except Exception as e:
        print(f"  ⚠️ Error extracting date: {e}")

    # Find "Key transactions" section
    key_transactions_section = None
    
    # Search for the heading
    for heading in soup.find_all(['h2', 'h3', 'h4', 'strong', 'b']):
        if 'key transaction' in heading.get_text().lower():
            key_transactions_section = heading
            print("  ✅ Found 'Key transactions' section")
            break
    
    if not key_transactions_section:
        print("  ⚠️ 'Key transactions' section not found, trying fallback extraction")
        # Fallback to old method
        full_text = soup.get_text()
        funding_patterns = [
            r'([A-Z][A-Za-z0-9\s&]+?)\s+(?:raised|secured|received)\s+(?:Rs\.?\s*)?(\d+(?:\.\d+)?\s*(?:crore|million|lakh))',
        ]

        for pattern in funding_patterns:
            matches = re.finditer(pattern, full_text)
            for match in matches:
                company_name = match.group(1).strip()
                funding_amount = match.group(2).strip()

                if len(company_name) > 50:
                    continue

                if not funding_amount.startswith('Rs') and not funding_amount.startswith('$'):
                    funding_amount = f"Rs {funding_amount}"

                country = extract_country_from_text(full_text, company_name)

                companies.append({
                    "Company_Name": company_name,
                    "Funding_Amount": funding_amount,
                    "Country": country,
                    "Founder_Name": "",
                    "Important_Person": "",
                    "Email": "",
                    "LinkedIn_URL": "",
                    "Industry_Sector": "",
                    "Category": "",
                    "Description": "",
                    "Date": article_date
                })

        return companies
    
    # Extract the content after "Key transactions" heading
    content_elements = []
    current = key_transactions_section.find_next_sibling()
    
    # Collect all paragraphs and divs until next major heading or end
    while current:
        if current.name in ['h2', 'h3'] and current != key_transactions_section:
            # Stop at next major heading
            break
        
        if current.name in ['p', 'div', 'ul', 'li']:
            text = current.get_text(strip=True)
            if len(text) > 20:  # Only substantial content
                content_elements.append(text)
        
        current = current.find_next_sibling()
    
    if not content_elements:
        print("  ⚠️ No content found in Key transactions section")
        return companies
    
    # Combine all content
    key_transactions_text = "\n\n".join(content_elements)
    
    print(f"  📝 Extracted Key transactions content ({len(key_transactions_text)} chars)")
    print(f"  🤖 Sending to {llm_name} AI for extraction...")
    
    # Use LLM to extract structured data
    try:
        prompt = f"""Extract ALL startup funding information from this "Key transactions" section. 
Each transaction is usually a separate paragraph or bullet point.

TEXT:
{key_transactions_text}

INSTRUCTIONS:
- Extract EVERY company/startup mentioned with funding details
- Company names are often highlighted or at the start of each transaction
- DO NOT include investor names, VC firms, or funding sources in any field
- Investors/VCs to IGNORE: Sequoia, Accel, Tiger Global, Peak XV, Lightspeed, Matrix Partners, Nexus VP, Kalaari, Blume, Elevation Capital, etc.
- For each company, extract ONLY:
  * company_name: The EXACT STARTUP/COMPANY name (e.g., "Zepto" not "Zeptor", "Kuku" not "Kukur") - DO NOT modify spelling
  * funding_amount: The amount raised - CRITICAL RULE:
    • If BOTH INR/Rs AND USD amounts are present (e.g., "Rs 160 crore ($18 million)"), extract ONLY the INR/Rs amount ("INR 160 Cr")
    • If only USD is present, extract USD amount
    • Always write rupee symbol as "INR" not "₹" or "Rs"
  * founder_name: ONLY actual human founder names (first and last name). Leave empty if not found.
  * industry: Industry/sector tags (fintech, AI, healthtech, SaaS, edtech, etc.)
  * description: Brief description of what the COMPANY does (not the investors)

CRITICAL: 
- Preserve EXACT company name spelling - do not change "Zepto" to "Zeptor" or "Kuku" to "Kukur"
- When both INR and USD amounts exist, ALWAYS prefer INR (e.g., "Rs 160 crore ($18 million)" → extract "INR 160 Cr")
- founder_name field should contain ONLY human names of founders/co-founders
- DO NOT put investor names, VC firms, or company names in founder_name field
- If no founder name is mentioned, leave founder_name empty
- Important_person should remain empty (we don't need investor data)

Return ONLY valid JSON array (no markdown, no code blocks):
[
  {{
    "company_name": "Exact startup name (not investor)",
    "funding_amount": "INR amount if both given, else $USD",
    "founder_name": "Human founder name only",
    "industry": "fintech, AI, etc",
    "description": "What the company does"
  }},
  ...
]"""

        # Use universal LLM caller
        response_text = call_llm(llm_name, current_api_key, prompt, max_tokens=1000)

        if not response_text:
            print(f"  ⚠️ {llm_name} API error, using fallback")
            rotate_api_key()
            response_text = call_llm(llm_name, current_api_key, prompt, max_tokens=1000)
            if not response_text:
                print(f"  ⚠️ {llm_name} API still failing after rotation, using fallback extraction")
                return companies

        # Clean response
        response_text = re.sub(r'```json\n?|\n?```', '', response_text)
        response_text = re.sub(r'```\n?', '', response_text)
        response_text = response_text.strip()

        try:
            extracted_companies = json.loads(response_text)
        except json.JSONDecodeError as e:
            print(f"  ⚠️ JSON parsing error: {e}")
            return companies

        if not isinstance(extracted_companies, list):
            print("  ⚠️ LLM returned non-list response")
            return companies

        print(f"  ✅ {llm_name} extracted {len(extracted_companies)} companies from Key transactions")

        # Known investor/VC keywords to filter out
        investor_keywords = [
            'capital', 'ventures', 'partners', 'fund', 'equity', 'investments',
            'venture', 'sequoia', 'accel', 'tiger', 'peak xv', 'lightspeed',
            'matrix', 'nexus', 'kalaari', 'blume', 'elevation', 'chiratae',
            'steadview', 'softbank', 'temasek', 'prosus', 'naspers'
        ]

        # Convert to our format
        full_text = soup.get_text()
        for comp in extracted_companies:
            company_name = comp.get('company_name', '').strip()
            funding_amount = comp.get('funding_amount', '').strip()
            founder_name = comp.get('founder_name', '').strip()
            
            if not company_name:
                continue
            
            # Validate founder name - check if it's actually an investor/VC firm
            if founder_name:
                founder_lower = founder_name.lower()
                is_investor = any(keyword in founder_lower for keyword in investor_keywords)
                
                # Also check if it's too similar to common VC patterns
                if is_investor or not is_valid_founder_name(founder_name):
                    print(f"  🚫 Filtered out investor/invalid name: {founder_name}")
                    founder_name = ""
                
                # Additional validation against company name
                if founder_name and not validate_founder_against_company(company_name, founder_name):
                    founder_name = ""
            
            # Convert funding to consistent format - prioritize INR
            if funding_amount:
                inr_amount = convert_to_inr(funding_amount)
            else:
                inr_amount = ""
            
            country = extract_country_from_text(full_text, company_name)

            companies.append({
                "Company_Name": company_name,
                "Funding_Amount": inr_amount if inr_amount else funding_amount,
                "Country": country,
                "Founder_Name": founder_name,  # Now properly filtered
                "Important_Person": "",  # Leave empty - no investor data needed
                "Email": "",
                "LinkedIn_URL": "",
                "Industry_Sector": comp.get('industry', '').strip(),
                "Category": "",
                "Description": comp.get('description', '').strip(),
                "Date": article_date  # ✅ Add extracted date
            })

        return companies

    except Exception as e:
        print(f"  ⚠️ Error using {llm_name} for YourStory: {e}")
        import traceback
        traceback.print_exc()
        return companies
    
def parse_entrackr(html, llm_name):
    """Parse Entrackr using LLM AI for 'Growth-stage deals' and 'Early-stage deals' sections."""
    soup = BeautifulSoup(html, "html.parser")
    companies = []

    # ✅ Extract date from page headline
    article_date = ""
    try:
        # Find the headline/title - try multiple selectors
        title_selectors = ['h1', 'h1.entry-title', '.headline', '.article-title', 'h1.post-title', '.page-title']
        for selector in title_selectors:
            title_element = soup.select_one(selector)
            if title_element:
                title_text = title_element.get_text().strip()
                print(f"  📰 Found headline: {title_text[:100]}...")
                # Extract date pattern like "Sept 20-26" or "October 20-26" or "this week"
                date_match = re.search(r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\s+\d{1,2}(?:-\d{1,2})?(?:,\s*\d{4})?', title_text, re.IGNORECASE)
                if date_match:
                    article_date = date_match.group(0)
                    print(f"  📅 Extracted date from headline: {article_date}")
                    break
                # Fallback: if headline contains "this week", use current date
                elif 'this week' in title_text.lower():
                    from datetime import datetime
                    article_date = datetime.now().strftime("%B %d, %Y")
                    print(f"  📅 Using current date: {article_date}")
                    break
        
        if not article_date:
            print(f"  ⚠️ Could not extract date from headline")
    except Exception as e:
        print(f"  ⚠️ Error extracting date: {e}")

    # Find "Growth-stage deals" and "Early-stage deals" sections
    sections_to_extract = ['Growth-stage deals', 'Early-stage deals']
    
    for section_name in sections_to_extract:
        print(f"\n  🔍 Looking for section: {section_name}")
        section_found = False
        
        # Search for the section heading (can be in <strong>, <span>, etc.)
        for element in soup.find_all(['strong', 'span', 'p', 'h2', 'h3']):
            element_text = element.get_text(strip=True)
            if section_name.lower() in element_text.lower():
                section_found = True
                print(f"  ✅ Found '{section_name}' section")
                
                # Extract content after this heading
                content_elements = []
                current = element.find_parent(['p', 'span']).find_next_sibling() if element.find_parent(['p', 'span']) else element.find_next_sibling()
                
                # Collect paragraphs until next major section or end
                while current:
                    current_text = current.get_text(strip=True)
                    
                    # Stop if we hit another section heading
                    if any(other_section in current_text for other_section in ['[Growth-stage deals]', '[Early-stage deals]', '[City and segment-wise deals]', 'City and segment-wise', 'For a detailed funding']):
                        break
                    
                    # Add substantial content
                    if current.name in ['p', 'span'] and len(current_text) > 50:
                        content_elements.append(current_text)
                    
                    current = current.find_next_sibling()
                
                if not content_elements:
                    print(f"  ⚠️ No content found in '{section_name}' section")
                    continue
                
                # Combine all content
                section_text = "\n\n".join(content_elements)
                
                print(f"  📝 Extracted {section_name} content ({len(section_text)} chars)")
                print(f"  🤖 Sending to {llm_name} AI for extraction...")
                
                # Use LLM to extract structured data
                try:
                    prompt = prompt = f"""Extract ALL startup funding information from this "{section_name}" section from Entrackr.

TEXT:
{section_text}

CRITICAL SPELLING RULE:
- Copy company names EXACTLY character-by-character as they appear in the text
- DO NOT fix spelling, DO NOT autocorrect, DO NOT change capitalization
- If text says "Uniphore", write "Uniphore" NOT "Uniphora"
- If text says "UnifyApps", write "UnifyApps" NOT "Unify Apps"
- PRESERVE EXACT SPELLING INCLUDING TYPOS

INSTRUCTIONS:
- Extract EVERY company/startup mentioned with funding details
- DO NOT include investor names, VC firms, or funding sources in any field
- Investors/VCs to IGNORE: NVIDIA, AMD, Snowflake, Databricks, WestBridge Capital, ICONIQ, Asha Ventures, British International Investment, IIFL Fintech Fund, Fashion Entrepreneur Fund, etc.
- For each company, extract ONLY:
  * company_name: The EXACT STARTUP/COMPANY name - COPY CHARACTER-BY-CHARACTER from text
    • "Uniphore" → "Uniphore" (NOT "Uniphora")
    • "UnifyApps" → "UnifyApps" (NOT "Unify Apps")
    • "Wonderland Foods" → "Wonderland Foods" (EXACT COPY)
  * funding_amount: The amount raised with currency (e.g., "$260 million", "$50 million", "Rs 140 crore")
    • Keep original currency format as mentioned in text
    • Write rupee symbol as "INR" or "Rs" based on text
  * funding_round: Series F, Series B, Pre-Series A, Seed, etc.
  * founder_name: ONLY actual human founder names (first and last name). Leave empty if not found.
  * industry: Industry/sector description if available (e.g., "conversational automation platform", "wastewater management", "trade financing platform")
    • Extract ONLY if explicitly mentioned in the text describing the company
    • If not mentioned, leave empty
  * description: Brief description of what the COMPANY does (only if available in text)

CRITICAL EXAMPLES FROM TEXT:
Example 1:
Text: "conversational automation platform Uniphore, which raised $260 million in a Series F round from NVIDIA"
→ company_name: "Uniphore" (EXACT COPY, not "Uniphora")
→ funding_amount: "$260 million"
→ funding_round: "Series F"
→ industry: "conversational automation platform"

Example 2:
Text: "UnifyApps secured $50 million in Series B funding"
→ company_name: "UnifyApps" (EXACT COPY, not "Unify Apps")
→ funding_amount: "$50 million"
→ funding_round: "Series B"

Example 3:
Text: "healthy-snacking brand Wonderland Foods raised Rs 140 crore"
→ company_name: "Wonderland Foods" (EXACT COPY)
→ funding_amount: "Rs 140 crore"
→ industry: "healthy-snacking brand"

CRITICAL RULES:
- NEVER alter company name spelling - copy it EXACTLY character-by-character
- When both INR and USD amounts exist, prefer the FIRST mentioned amount
- founder_name field should contain ONLY human names of founders/co-founders
- DO NOT put investor names, VC firms, or funding sources in founder_name or important_person
- Industry should be the descriptive term BEFORE the company name
- If no industry description is given, leave it empty

Return ONLY valid JSON array (no markdown, no code blocks):
[
  {{
    "company_name": "EXACT company name from text",
    "funding_amount": "Amount with currency as mentioned",
    "funding_round": "Round type if mentioned",
    "founder_name": "Human founder name only",
    "industry": "Industry description if available",
    "description": "Brief description if available"
  }},
  ...
]"""
                    # Use universal LLM caller
                    response_text = call_llm(llm_name, current_api_key, prompt, max_tokens=1500)

                    if not response_text:
                        print(f"  ⚠️ {llm_name} API error for {section_name}, using fallback")
                        rotate_api_key()
                        response_text = call_llm(llm_name, current_api_key, prompt, max_tokens=1500)
                        if not response_text:
                            print(f"  ⚠️ {llm_name} API still failing after rotation for {section_name}")
                            continue

                    # Clean response
                    response_text = re.sub(r'```json\n?|\n?```', '', response_text)
                    response_text = re.sub(r'```\n?', '', response_text)
                    response_text = response_text.strip()

                    try:
                        extracted_companies = json.loads(response_text)
                    except json.JSONDecodeError as e:
                        print(f"  ⚠️ JSON parsing error for {section_name}: {e}")
                        continue

                    if not isinstance(extracted_companies, list):
                        print(f"  ⚠️ LLM returned non-list response for {section_name}")
                        continue

                    print(f"  ✅ {llm_name} extracted {len(extracted_companies)} companies from {section_name}")

                    # Known investor/VC keywords to filter out
                    investor_keywords = [
                        'capital', 'ventures', 'partners', 'fund', 'equity', 'investments',
                        'venture', 'sequoia', 'accel', 'tiger', 'peak xv', 'lightspeed',
                        'matrix', 'nexus', 'kalaari', 'blume', 'elevation', 'chiratae',
                        'steadview', 'softbank', 'temasek', 'prosus', 'naspers', 'nvidia',
                        'amd', 'snowflake', 'databricks', 'westbridge', 'iconiq', 'asha',
                        'british international', 'iifl', 'fashion entrepreneur'
                    ]

                    # Convert to our format
                    full_text = soup.get_text()
                    for comp in extracted_companies:
                        company_name = comp.get('company_name', '').strip()
                        funding_amount = comp.get('funding_amount', '').strip()
                        funding_round = comp.get('funding_round', '').strip()
                        founder_name = comp.get('founder_name', '').strip()
                        industry = comp.get('industry', '').strip()
                        
                        if not company_name:
                            continue
                        
                        # Validate founder name - check if it's actually an investor/VC firm
                        if founder_name:
                            founder_lower = founder_name.lower()
                            is_investor = any(keyword in founder_lower for keyword in investor_keywords)
                            
                            if is_investor or not is_valid_founder_name(founder_name):
                                print(f"  🚫 Filtered out investor/invalid name: {founder_name}")
                                founder_name = ""
                            
                            # Additional validation against company name
                            if founder_name and not validate_founder_against_company(company_name, founder_name):
                                founder_name = ""
                        
                        # Convert funding to consistent format if needed
                        if funding_amount:
                            inr_amount = convert_to_inr(funding_amount)
                        else:
                            inr_amount = ""

                        companies.append({
                            "Company_Name": company_name,
                            "Funding_Amount": inr_amount if inr_amount else funding_amount,
                            "Country": "India",  # Set India for all Entrackr companies
                            "Founder_Name": founder_name,
                            "Important_Person": "",
                            "Email": "",
                            "LinkedIn_URL": "",
                            "Industry_Sector": industry,
                            "Category": "",
                            "Description": comp.get('description', '').strip(),
                            "Date": article_date
                        })

                except Exception as e:
                    print(f"  ⚠️ Error using {llm_name} for {section_name}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
                
                break  # Found the section, move to next
        
        if not section_found:
            print(f"  ⚠️ '{section_name}' section not found")

    print(f"  📈 Entrackr extracted {len(companies)} companies total")
    print(f"  ✅ {llm_name} extracted {len(extracted_companies)} companies from {section_name}")

# ✅ ADD THIS DEBUG SECTION
    print(f"\n  🔍 DEBUG: Companies extracted from {section_name}:")
    for idx, comp in enumerate(extracted_companies, 1):
     print(f"    {idx}. {comp.get('company_name', 'N/A')} - {comp.get('funding_amount', 'N/A')}")
    return companies

def format_inc42_date(row_date, article_date_context=""):
    """
    Format Inc42 row date properly.
    Inc42 dates are like "Oct 28" or "28 Oct" - we need to add the year.
    
    Args:
        row_date: Date from table row (e.g., "Oct 28", "28 Oct", "28/10")
        article_date_context: Article date with year for context
    
    Returns:
        Formatted date string with year
    """
    from datetime import datetime
    import re
    
    if not row_date or row_date.strip() == '':
        return ''
    
    row_date = row_date.strip()
    
    # Get current year as default
    current_year = datetime.now().year
    
    # Try to extract year from article context if available
    context_year = current_year
    if article_date_context:
        year_match = re.search(r'\b(20\d{2})\b', article_date_context)
        if year_match:
            context_year = int(year_match.group(1))
    
    # Pattern 1: "Oct 28" or "OCT 28"
    if re.match(r'^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\s+\d{1,2}$', row_date, re.IGNORECASE):
        return f"{row_date}, {context_year}"
    
    # Pattern 2: "28 Oct" or "28 OCT"
    match = re.match(r'^(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)$', row_date, re.IGNORECASE)
    if match:
        day = match.group(1)
        month = match.group(2)
        return f"{month} {day}, {context_year}"
    
    # Pattern 3: "28/10" or "28-10"
    match = re.match(r'^(\d{1,2})[/-](\d{1,2})$', row_date)
    if match:
        day = match.group(1)
        month = match.group(2)
        try:
            date_obj = datetime(context_year, int(month), int(day))
            return date_obj.strftime('%b %d, %Y')
        except:
            pass
    
    # Pattern 4: Already has year "Oct 28, 2025"
    if re.search(r'\b20\d{2}\b', row_date):
        return row_date
    
    # Fallback: return as-is with year added
    return f"{row_date}, {context_year}"
def parse_inc42(html, llm_name):
    """Parse Inc42 funding galore table."""
    from bs4 import BeautifulSoup
    import re
    
    soup = BeautifulSoup(html, "html.parser")
    companies = []

    # ✅ Extract article date from page headline/title for context
    article_date_context = ""
    try:
        # Try multiple selectors for title
        title_selectors = ['h1', 'h1.entry-title', '.headline', '.article-title', 'title']
        for selector in title_selectors:
            title_element = soup.select_one(selector)
            if title_element:
                title_text = title_element.get_text().strip()
                print(f"  📰 Found headline: {title_text[:100]}...")
                
                # Extract date pattern like "Oct 20, 2025" or "October 20-26, 2025"
                date_match = re.search(r'\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+(\d{1,2})(?:-(\d{1,2}))?,?\s*(\d{4})\b', title_text, re.IGNORECASE)
                
                if date_match:
                    month = date_match.group(1)
                    start_day = date_match.group(2)
                    end_day = date_match.group(3)
                    year = date_match.group(4)
                    
                    if end_day:
                        article_date_context = f"{month} {start_day}-{end_day}, {year}"
                    else:
                        article_date_context = f"{month} {start_day}, {year}"
                    
                    print(f"  📅 Article context date: {article_date_context}")
                    break
        
        if not article_date_context:
            print(f"  ℹ️ No article date found in headline (will use row dates)")
    except Exception as e:
        print(f"  ⚠️ Error extracting article date: {e}")

    # Find all tables
    tables = soup.find_all('table')
    
    if not tables:
        print("  ❌ No HTML tables found on the page")
        return companies
    
    print(f"  📊 Processing {len(tables)} table(s)...")
    
    for table_idx, table in enumerate(tables):
        print(f"\n  📋 Analyzing Table {table_idx + 1}...")
        
        # Get table headers
        headers = []
        thead = table.find('thead')
        if thead:
            header_row = thead.find('tr')
            if header_row:
                headers = [clean_text(th.get_text(strip=True)) for th in header_row.find_all(['th', 'td'])]
        
        print(f"    Headers found: {headers}")
        
        # Find tbody rows
        tbody = table.find('tbody')
        if not tbody:
            print("    ⚠️ No tbody found, skipping table")
            continue
        
        rows = tbody.find_all('tr')
        print(f"    Found {len(rows)} data rows")
        
        for row_idx, row in enumerate(rows):
            try:
                cells = row.find_all(['td', 'th'])
                
                if len(cells) < 2:
                    continue
                
                # Extract cell data and clean text
                cell_data = [clean_text(cell.get_text(strip=True)) for cell in cells]
                
                # Skip if row looks like a header
                if any(keyword in ' '.join(cell_data).lower() for keyword in ['date', 'name', 'sector', 'subsector']) and row_idx <= 1:
                    continue
                
                # Initialize
                category = ''
                funding_amount = ''
                row_date = ''  # ✅ Use date from row
                
                # Extract based on number of columns
                if len(cell_data) >= 4:
                    # Structure: Date | Name | Sector | Subsector | Category | Amount | Round | Investors | Lead
                    row_date = cell_data[0]  # ✅ FIRST COLUMN IS THE DATE
                    company_name = cell_data[1]
                    industry_sector = cell_data[2]
                    
                    # Cell[4] = Category (B2B/B2C)
                    # Cell[5] = Funding Amount
                    if len(cell_data) > 4:
                        category = cell_data[4] if cell_data[4] in ['B2B', 'B2C', 'B2B2C', 'D2C', 'C2C'] else ''
                    
                    if len(cell_data) > 5:
                        funding_amount = clean_text(cell_data[5])
                    
                    # ✅ Format the date properly (add year if missing)
                    formatted_date = format_inc42_date(row_date, article_date_context)
                    
                    # Create entry
                    funding_entry = {
                        'Company_Name': company_name,
                        'Funding_Amount': funding_amount,
                        'Country': 'India',
                        'Founder_Name': '',
                        'Important_Person': '',
                        'Email': '',
                        'LinkedIn_URL': '',
                        'Industry_Sector': industry_sector,
                        'Category': category,
                        'Description': '',
                        'Date': formatted_date  # ✅ Use row date
                    }
                
                elif len(cell_data) == 3:
                    # Date | Name | Sector
                    row_date = cell_data[0]
                    formatted_date = format_inc42_date(row_date, article_date_context)
                    
                    funding_entry = {
                        'Company_Name': cell_data[1],
                        'Funding_Amount': '',
                        'Country': 'India',
                        'Founder_Name': '',
                        'Important_Person': '',
                        'Email': '',
                        'LinkedIn_URL': '',
                        'Industry_Sector': cell_data[2],
                        'Category': '',
                        'Description': '',
                        'Date': formatted_date
                    }
                
                elif len(cell_data) == 2:
                    # Date | Name
                    row_date = cell_data[0]
                    formatted_date = format_inc42_date(row_date, article_date_context)
                    
                    funding_entry = {
                        'Company_Name': cell_data[1],
                        'Funding_Amount': '',
                        'Country': 'India',
                        'Founder_Name': '',
                        'Important_Person': '',
                        'Email': '',
                        'LinkedIn_URL': '',
                        'Industry_Sector': '',
                        'Category': '',
                        'Description': '',
                        'Date': formatted_date
                    }
                
                # Validate company name
                company_name = funding_entry.get('Company_Name', '').strip()
                
                # Skip footer text
                if any(skip_text in company_name.lower() for skip_text in [
                    'part of a larger round',
                    'source:',
                    'note:',
                    'only disclosed',
                    'inc42'
                ]):
                    continue
                
                if funding_entry and company_name and len(company_name) > 1:
                    companies.append(funding_entry)
                    print(f"    ✅ [{row_idx}] {company_name} | {funding_entry['Date']} | {funding_entry['Category']} | {funding_entry['Funding_Amount']}")
                
            except Exception as e:
                print(f"    ⚠️ Error parsing row {row_idx}: {str(e)[:100]}")
                continue
    
    print(f"  📈 Inc42 extracted {len(companies)} companies")
    return companies
def parse_alleywatch(html):
    """Parse AlleyWatch."""
    soup = BeautifulSoup(html, "html.parser")
    companies = []

    # ✅ Extract date from page headline
    article_date = ""
    try:
        # Find the headline/title - try multiple selectors
        title_selectors = ['h1', 'h1.entry-title', '.headline', '.article-title', 'h1.post-title', '.page-title']
        for selector in title_selectors:
            title_element = soup.select_one(selector)
            if title_element:
                title_text = title_element.get_text().strip()
                print(f"  📰 Found headline: {title_text[:100]}...")
                # Extract date pattern like "10/23/2025" or "10/23/25"
                date_match = re.search(r'(\d{1,2}/\d{1,2}/\d{2,4})', title_text)
                if date_match:
                    extracted_date = date_match.group(1)
                    article_date = f"Reported on {extracted_date}"
                    print(f"  📅 Extracted date from headline: {article_date}")
                    break
        
        if not article_date:
            print(f"  ⚠️ Could not extract date from headline")
    except Exception as e:
        print(f"  ⚠️ Error extracting date: {e}")

    full_text = soup.get_text()
    
    # Try multiple selectors for AlleyWatch's structure
    headings = soup.find_all(['h2', 'h3', 'h4'])
    
    # Also try to find list items or paragraphs with funding info
    for heading in headings:
        heading_text = heading.get_text(strip=True)
        
        # Pattern 1: Company Name - $Amount
        match = re.match(r'^(.+?)\s*[–—-]\s*\$?([\d.]+[MBK]?M?)$', heading_text, re.IGNORECASE)
        
        if match:
            company_name = match.group(1).strip()
            funding_amount = match.group(2).strip()

            if not funding_amount.startswith('$'):
                funding_amount = f"${funding_amount}"

            country = extract_country_from_text(full_text, company_name)

            companies.append({
                "Company_Name": company_name,
                "Funding_Amount": funding_amount,
                "Country": country,
                "Founder_Name": "",
                "Important_Person": "",
                "Email": "",
                "LinkedIn_URL": "",
                "Industry_Sector": "",
                "Category": "",
                "Description": "",
                "Date": article_date  # ✅ Add extracted date
            })
    
    # Fallback: Look for paragraphs with funding patterns
    if not companies:
        paragraphs = soup.find_all('p')
        for p in paragraphs:
            text = p.get_text(strip=True)
            # Match: Company raised $Amount or Company - $Amount
            matches = re.finditer(r'([A-Z][A-Za-z0-9\s&]+?)\s+(?:raised|secured|–|—|-)\s+\$?([\d.]+[MBK]?M?)', text, re.IGNORECASE)
            for match in matches:
                company_name = match.group(1).strip()
                funding_amount = match.group(2).strip()
                
                if len(company_name) > 50:
                    continue
                
                if not funding_amount.startswith('$'):
                    funding_amount = f"${funding_amount}"
                
                country = extract_country_from_text(full_text, company_name)
                
                companies.append({
                    "Company_Name": company_name,
                    "Funding_Amount": funding_amount,
                    "Country": country,
                    "Founder_Name": "",
                    "Important_Person": "",
                    "Email": "",
                    "LinkedIn_URL": "",
                    "Industry_Sector": "",
                    "Category": "",
                    "Description": "",
                    "Date": article_date  # ✅ Add extracted date
                })

    print(f"  📈 AlleyWatch extracted {len(companies)} companies")
    return companies

def parse_crunchbase(html):
    """Parse Crunchbase News."""
    soup = BeautifulSoup(html, "html.parser")
    companies = []

    # ✅ Extract date from page metadata/script (from Crunchbase News HTML)
    article_date = ""
    try:
        # Method 1: Look for pagePostDate in script tag (as shown in your screenshot)
        script_tags = soup.find_all('script')
        for script in script_tags:
            script_text = script.string if script.string else ""
            # Search for pattern like "pagePostDate":"October 24, 2025"
            date_match = re.search(r'"pagePostDate"\s*:\s*"([^"]+)"', script_text)
            if date_match:
                extracted_date = date_match.group(1)
                article_date = f"Reported on {extracted_date}"
                print(f"  📅 Extracted date from script: {article_date}")
                break
        
        # Method 2: Fallback - look for date in meta tags
        if not article_date:
            meta_date = soup.find('meta', {'property': 'article:published_time'})
            if meta_date and meta_date.get('content'):
                date_str = meta_date.get('content')
                # Parse ISO date format and convert to readable format
                try:
                    from datetime import datetime
                    parsed_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                    formatted_date = parsed_date.strftime('%B %d, %Y')
                    article_date = f"Reported on {formatted_date}"
                    print(f"  📅 Extracted date from meta tag: {article_date}")
                except:
                    pass
        
        # Method 3: Look for date in time tags
        if not article_date:
            time_tag = soup.find('time', {'datetime': True})
            if time_tag:
                date_str = time_tag.get('datetime')
                try:
                    from datetime import datetime
                    parsed_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                    formatted_date = parsed_date.strftime('%B %d, %Y')
                    article_date = f"Reported on {formatted_date}"
                    print(f"  📅 Extracted date from time tag: {article_date}")
                except:
                    pass
        
        if not article_date:
            print(f"  ⚠️ Could not extract date from Crunchbase page")
    except Exception as e:
        print(f"  ⚠️ Error extracting date: {e}")

    full_text = soup.get_text()
    paragraphs = soup.find_all('p')

    for p in paragraphs:
        text = p.get_text(strip=True)
        # Pattern: "1. Company Name, $50M" or "1. (tied) Company Name, $50M"
        match = re.match(r'^\d+\.\s*(?:\(tied\)\s*)?([^,]+),\s*\$?([\d.]+[MBK]?M?)', text, re.IGNORECASE)

        if match:
            company_name = match.group(1).strip()
            funding_amount = match.group(2).strip()

            if not funding_amount.startswith('$'):
                funding_amount = f"${funding_amount}"

            country = extract_country_from_text(text, company_name)

            companies.append({
                "Company_Name": company_name,
                "Funding_Amount": funding_amount,
                "Country": country,
                "Founder_Name": "",
                "Important_Person": "",
                "Email": "",
                "LinkedIn_URL": "",
                "Industry_Sector": "",
                "Category": "",
                "Description": "",
                "Date": article_date  # ✅ Add extracted date
            })

    print(f"  📈 Crunchbase extracted {len(companies)} companies")
    return companies
# ==================== APPEND TO CSV FUNCTION ====================
def append_companies_to_csv(new_companies, csv_path):
    """
    Append new companies to existing CSV with TWO-STAGE deduplication:
    1. First round: Remove duplicates by company name
    2. Second round: Remove duplicates by founder name
    """
    if not new_companies:
        return 0

    # Load existing CSV
    try:
        existing_df = pd.read_csv(csv_path, encoding='utf-8-sig')
        existing_companies = set(existing_df['Company_Name'].str.lower().str.strip())
        # Extract existing founder names (non-empty)
        existing_founders = set(
            existing_df[existing_df['Founder_Name'].str.len() > 0]['Founder_Name']
            .str.lower()
            .str.strip()
        )
    except:
        existing_companies = set()
        existing_founders = set()

    print(f"\n🔍 DEDUPLICATION PROCESS:")
    print(f"  • Existing companies in CSV: {len(existing_companies)}")
    print(f"  • Existing founders in CSV: {len(existing_founders)}")
    print(f"  • New companies to process: {len(new_companies)}")

    # ========================================
    # ROUND 1: DEDUPLICATE BY COMPANY NAME
    # ========================================
    round1_unique = []
    round1_duplicates = []
    
    for company in new_companies:
        company_name_lower = company['Company_Name'].lower().strip()
        
        if company_name_lower not in existing_companies:
            # Validate founder names against company name
            company_name = company.get('Company_Name', '')
            founder_name = company.get('Founder_Name', '')
            important_person = company.get('Important_Person', '')

            # Check founder_name
            if founder_name and not validate_founder_against_company(company_name, founder_name):
                company['Founder_Name'] = ""

            # Check important_person
            if important_person:
                persons = [p.strip() for p in important_person.split(",")]
                valid_persons = [p for p in persons if validate_founder_against_company(company_name, p)]
                company['Important_Person'] = ", ".join(valid_persons)

            round1_unique.append(company)
            existing_companies.add(company_name_lower)
        else:
            round1_duplicates.append(company_name_lower)
    
    print(f"\n✅ ROUND 1 (Company Name Deduplication):")
    print(f"  • Passed: {len(round1_unique)} companies")
    print(f"  • Filtered out: {len(round1_duplicates)} duplicates")
    if round1_duplicates[:3]:
        print(f"  • Sample duplicates: {', '.join(list(round1_duplicates)[:3])}")

    # ========================================
    # ROUND 2: DEDUPLICATE BY FOUNDER NAME
    # ========================================
    round2_unique = []
    round2_duplicates = []
    
    for company in round1_unique:
        founder_name = company.get('Founder_Name', '').strip()
        
        # Skip founder deduplication if founder name is empty
        if not founder_name:
            round2_unique.append(company)
            continue
        
        founder_name_lower = founder_name.lower().strip()
        
        # Check if this founder already exists
        if founder_name_lower not in existing_founders:
            round2_unique.append(company)
            existing_founders.add(founder_name_lower)
        else:
            round2_duplicates.append({
                'company': company['Company_Name'],
                'founder': founder_name
            })
    
    print(f"\n✅ ROUND 2 (Founder Name Deduplication):")
    print(f"  • Passed: {len(round2_unique)} companies")
    print(f"  • Filtered out: {len(round2_duplicates)} duplicates")
    if round2_duplicates[:3]:
        print(f"  • Sample duplicates:")
        for dup in round2_duplicates[:3]:
            print(f"    - {dup['company']} (founder: {dup['founder']})")

    # ========================================
    # SAVE FINAL UNIQUE COMPANIES
    # ========================================
    if round2_unique:
        # Append to CSV
        new_df = pd.DataFrame(round2_unique)
        new_df.to_csv(csv_path, mode='a', header=False, index=False, encoding='utf-8-sig')
        
        print(f"\n🎉 DEDUPLICATION COMPLETE:")
        print(f"  • Total new companies added: {len(round2_unique)}")
        print(f"  • Total filtered (both rounds): {len(new_companies) - len(round2_unique)}")
        
        return len(round2_unique)
    else:
        print(f"\n⚠️ No new unique companies to add (all filtered as duplicates)")
        return 0

def scrape_other_sources(timeframe, region, output_csv, llm_name):
    """Phase 2: Scrape other sources and append to existing CSV."""
    print("\n" + "=" * 70)
    print(f"🚀 PHASE 2: Other Sources Scraper")
    print(f"Timeframe: {timeframe} | Region: {region}")
    print("=" * 70)

    total_added = 0

    indian_sources = [
        {
            'name': 'YOURSTORY',
            'query': 'Weekly funding roundup yourstory',
            'use_directory': True,  # Flag to use directory extraction
            'parser': lambda html: parse_yourstory(html, llm_name)
        },
        {
            'name': 'ENTRACKR',
            'query': 'entrackr weekly funding report 2025',
            'use_directory': True,  # Flag to use directory extraction
            'parser': lambda html: parse_entrackr(html, llm_name)
        },
        {
            'name': 'INC42',
            'query': 'funding galore inc42',
            'use_directory': True,  # ✅ CHANGED TO TRUE
            'parser': lambda html: parse_inc42(html, llm_name)
        }
    ]

    international_sources = [
        {
            'name': 'CRUNCHBASE NEWS',
            'query': f"site:news.crunchbase.com last week's biggest funding rounds",
            'parser': parse_crunchbase
        },
        {
            'name': 'ALLEYWATCH',
            'query': f"alleywatch weekly funding report {timeframe}",
            'parser': parse_alleywatch
        }
    ]

    if region.lower() in ['india', 'indian']:
        sources = indian_sources
        print(f"REGION: {region.upper()} - Using Indian sources only")
    else:
        sources = indian_sources + international_sources
        print(f"REGION: {region.upper()} - Using all sources")

    print(f"Sources to scrape: {len(sources)}")

    for idx, source in enumerate(sources, 1):
        print(f"\n[{idx}/{len(sources)}] {source['name']}")
        print("-" * 70)
        print(f"Searching: {source['query']}")

        # Check if we need to use directory-based extraction
        use_directory = source.get('use_directory', False)
        
        if use_directory:
            # Step 1: Search for directory page
            directory_link = search_duckduckgo_selenium(source['query'])
            
            if not directory_link:
                print(f"  ❌ Could not find directory page for {source['name']}")
                continue
            
            print(f"  📂 Found directory page: {directory_link}")
            
            # Step 2: Extract latest article link from directory
            article_link = get_latest_article_link(directory_link, source['name'])
            
            if not article_link:
                print(f"  ❌ Could not extract latest article from directory")
                continue
            
            link = article_link
            print(f"  📰 Latest article URL extracted: {link}")
            
            # ✅ CRITICAL: Add delay and verify link before scraping
            time.sleep(1)
            print(f"  🌐 Now opening article page for scraping...")
        else:
            # Original method: direct search
            link = search_duckduckgo_selenium(source['query'])

        if link:
            print(f"  🔗 Scraping URL: {link}") 
            try:
                html = scrape_with_fallback(link)
                if not html:
                    print(f"  ❌ Failed to scrape {source['name']}")
                    continue

                companies = source['parser'](html)

                # Filter by region if needed
                filtered_companies = []
                for company_data in companies:
                    country = company_data.get('Country', '')

                    if region.lower() in ['india', 'indian']:
                        if isinstance(country, str) and ('india' in country.lower()):
                            filtered_companies.append(company_data)
                    else:
                        filtered_companies.append(company_data)

                # Append to CSV (includes founder validation)
                added = append_companies_to_csv(filtered_companies, output_csv)
                total_added += added
                print(f"✓ {source['name']}: Processed {len(filtered_companies)} companies")

            except Exception as e:
                print(f"✗ Error scraping {source['name']}: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"✗ No link found for {source['name']}")

        # Longer delays for better stealth
        delay = random.uniform(3,4)
        print(f"⏳ Waiting {delay:.1f}s before next source...")
        sleep(delay)

    print(f"\n✅ Phase 2 Complete: Added {total_added} new companies from other sources")
    cleanup_selenium_driver()
    return total_added

def test_network_connectivity():
    """Test if we can reach required services"""
    import socket
    
    print("\n🔍 Testing Network Connectivity...")
    
    services = {
        "DuckDuckGo": "duckduckgo.com",
        "Mistral API": "api.mistral.ai",
        "Claude API": "api.anthropic.com",
        "Apollo API": "api.apollo.io"
    }
    
    failures = []
    for service, domain in services.items():
        try:
            socket.gethostbyname(domain)
            print(f"  ✅ {service}: {domain} reachable")
        except socket.gaierror:
            print(f"  ❌ {service}: {domain} FAILED")
            failures.append(service)
    
    if failures:
        print(f"\n⚠️ WARNING: Cannot reach {', '.join(failures)}")
        print("  This may cause enrichment to fail.")
        return False
    
    print(f"✅ All services reachable")
    return True   

def sort_by_date(df):
    """
    Sort DataFrame by Date column with latest dates first.
    Handles multiple date formats intelligently.
    """
    from datetime import datetime
    import pandas as pd
    
    print("\n📅 SORTING DATA BY DATE (LATEST FIRST)...")
    
    # Create a copy to avoid modifying original
    df_sorted = df.copy()
    
    # Function to parse various date formats
    def parse_date(date_str):
        """Parse date string into datetime object."""
        if pd.isna(date_str) or not date_str or date_str.strip() == '':
            return datetime.min  # Put empty dates at the end
        
        date_str = str(date_str).strip()
        
        # Remove "Reported on" prefix if exists
        date_str = date_str.replace('Reported on ', '')
        
        # ✅ ADD CURRENT YEAR TO DATES WITHOUT YEAR
        current_year = datetime.now().year
        
        # Check if date has no year (e.g., "Oct 25" or "OCT 25")
        # Pattern: Month Day (no year)

        # Check if date has no year (e.g., "Oct 25" or "OCT 25" or "6 Nov")
        # Pattern: Month Day (no year) OR Day Month (no year)
        no_year_pattern_1 = r'^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\s+(\d{1,2})$'
        no_year_pattern_2 = r'^(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)$'
        
        if re.match(no_year_pattern_1, date_str, re.IGNORECASE):
            date_str = f"{date_str}, {current_year}"
            print(f"  ℹ️ Added current year to date: {date_str}")
        elif re.match(no_year_pattern_2, date_str, re.IGNORECASE):
            date_str = f"{date_str} {current_year}"
            print(f"  ℹ️ Added current year to date: {date_str}")
        
        # Try various formats
        # Try various formats
        formats = [
            '%d %b %Y',            # 6 Nov 2025 (NEW - ADD THIS FIRST)
            '%d %B %Y',            # 6 November 2025 (NEW)
            '%B %d, %Y',           # October 20, 2025
            '%b %d, %Y',           # Oct 20, 2025
            '%B %d-%d, %Y',        # October 20-26, 2025
            '%b %d-%d, %Y',        # Oct 20-26, 2025
            '%m/%d/%Y',            # 10/20/2025
            '%m/%d/%y',            # 10/20/25
            '%Y-%m-%d',            # 2025-10-20
            '%d-%m-%Y',            # 20-10-2025
            '%d/%m/%Y',            # 20/10/2025
        ]
        
        # For date ranges, use the start date
        if '-' in date_str and not date_str.count('-') > 2:
            # Handle "Oct 20-26, 2025" format
            parts = date_str.split('-')
            if len(parts) == 2:
                date_str = parts[0].strip()  # Use start date
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except:
                continue
        
        # If all parsing fails, return minimum date
        print(f"  ⚠️ Could not parse date: '{date_str}'")
        return datetime.min
    
    # Create sortable date column
    df_sorted['_sort_date'] = df_sorted['Date'].apply(parse_date)
    
    # Sort by date (descending - latest first)
    df_sorted = df_sorted.sort_values('_sort_date', ascending=False)
    
    # Remove temporary sorting column
    df_sorted = df_sorted.drop('_sort_date', axis=1)
    
    # Reset index
    df_sorted = df_sorted.reset_index(drop=True)
    
    # Show date distribution
    print(f"\n  ✅ Sorted {len(df_sorted)} companies by date")
    if 'Date' in df_sorted.columns:
        date_counts = df_sorted['Date'].value_counts().head(10)
        print(f"\n  📊 Top dates (latest first):")
        for date_val, count in date_counts.items():
            if date_val:
                print(f"    • {date_val}: {count} companies")
    
    return df_sorted

def process_leads(region, api_key_1, api_key_2, llm_name):  # ✅ Changed parameter names
    """
    Main function to process leads based on region with API key rotation.
    """
    
    # ✅ ADD VALIDATION BEFORE INITIALIZING
    print("\n🔍 VALIDATING RECEIVED API KEYS:")
    print(f"  • api_key_1: {api_key_1[:10] if api_key_1 else '❌ EMPTY'}...")  # ✅ Changed
    print(f"  • api_key_2: {api_key_2[:10] if api_key_2 else '⚠️ EMPTY'}...")  # ✅ Changed
    print(f"  • llm_name: {llm_name}")
    
    if not api_key_1:  # ✅ Changed
        print("❌ CRITICAL ERROR: Primary API key is empty!")
        raise ValueError("Primary API key (api_key_1) cannot be empty")  # ✅ Changed

    # Initialize API keys
    try:
        initialize_api_keys(api_key_1, api_key_2)  # ✅ Changed
    except Exception as e:
        print(f"❌ Failed to initialize API keys: {e}")
        raise

    timeframe = this_month
    output_csv = f"funding_data_{region}_{timeframe.replace(' ', '_').replace('/', '_')}.csv"

    print("\n" + "=" * 80)
    print("🎯 PROCESS LEADS - INTEGRATED FUNDING DATA SCRAPER")
    print("=" * 80)
    print(f"📅 Timeframe: {timeframe}")
    print(f"🌍 Region: {region}")
    print(f"🔑 API Keys: Initialized with rotation support")
    print(f"📁 Output File: {output_csv}")
    print("=" * 80)
    print("\n📋 SCRAPING STRATEGY:")
    print("  1️⃣  Phase 1: FoundersDay (Mistral AI) - Creates initial CSV")
    print("  2️⃣  Phase 2: Other sources - Append to CSV")
    if region.lower() in ['india', 'indian']:
        print("  🇮🇳  Indian region: Using YourStory + StartupTalky")
    else:
        print("  🌍  Global region: Using all sources (YourStory, StartupTalky, Crunchbase, AlleyWatch)")
    print("  🚫  Company-Founder validation: Remove founders matching company names")
    print("=" * 80)

    try:
        # PHASE 1: FoundersDay scraping (creates CSV)
        foundersday_count = scrape_foundersday_with_mistral(output_csv,llm_name)

        # PHASE 2: Other sources scraping (appends to CSV)  
        other_sources_count = scrape_other_sources(timeframe, region, output_csv,llm_name)

        # Final summary
        try:
            final_df = pd.read_csv(output_csv)
            final_df = sort_by_date(final_df)
            final_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
            print(f"  💾 Saved sorted data to: {output_csv}")
            total_companies = len(final_df)

            print("\n" + "=" * 80)
            print("✅ PROCESS LEADS COMPLETED SUCCESSFULLY!")
            print("=" * 80)
            print(f"\n📊 FINAL STATISTICS:")
            print(f"  • FoundersDay Companies: {foundersday_count}")
            print(f"  • Other Sources Added: {other_sources_count}")
            print(f"  • Total Unique Companies: {total_companies}")
            print(f"\n📁 Output CSV: {output_csv}")

            print("\n🌍 Country Distribution:")
            print(final_df['Country'].value_counts())

            print("\n🏭 Companies with Industry Info:")
            industry_count = len(final_df[final_df['Industry_Sector'].str.len() > 0])
            print(f"  {industry_count} companies ({industry_count/total_companies*100:.1f}%)")

            print("\n👥 Companies with Founder Info:")
            founder_count = len(final_df[final_df['Founder_Name'].str.len() > 0])
            print(f"  {founder_count} companies ({founder_count/total_companies*100:.1f}%)")

            print("\n🚫 Founder-Company Name Validation:")
            print("  Companies with founder names matching company names were filtered out")

            print("\n" + "=" * 80)
            print("🎉 CSV IS READY FOR USE!")
            print("=" * 80)

            # ============================================================
            # PHASE 3: ENRICH DATA WITH FOUNDER/CEO INFO
            # ============================================================
            print("\n" + "=" * 80)
            print("🔍 PHASE 3: ENRICHING DATA WITH FOUNDER/CEO INFORMATION")
            print("=" * 80)
            if not test_network_connectivity():
              print("\n⚠️ Network connectivity issues detected")
              print("   Skipping enrichment - returning raw scraped data")
              return final_df, output_csv
            try:
                from selenium_extractor import extract_company_data
                
                # Generate enriched output filename
                enriched_output = output_csv.replace('.csv', '_enriched.xlsx')
                
                print(f"\n🤖 Starting enrichment process...")
                print(f"  Input: {len(final_df)} companies")
                print(f"  Output will be saved to: {enriched_output}")
                import socket
                original_timeout = socket.getdefaulttimeout()
                socket.setdefaulttimeout(30)  
                # Call enrichment with the scraped DataFrame
                try : 
                 enriched_df, enriched_file = extract_company_data(
    csv_path=final_df,
    llm_provider=llm_name,
    llm_api_keys=[api_key_1, api_key_2],
    output_path=enriched_output,
    company_column='Company_Name',
    apollo_api_key=None,
   
)               
                finally:
                    socket.setdefaulttimeout(original_timeout)
                
                if enriched_df is not None and not enriched_df.empty:
                    enriched_df = sort_by_date(enriched_df)
                    enriched_df.to_excel(enriched_file, index=False, engine='openpyxl')
                    print(f"  💾 Saved sorted enriched data to: {enriched_file}")
                    print(f"\n✅ ENRICHMENT SUCCESSFUL!")
                    print(f"  • Enriched companies: {len(enriched_df)}")
                    print(f"  • Enriched file: {enriched_file}")
                    
                    # Return enriched data
                    return enriched_df, enriched_file
                else:
                    print(f"\n⚠️ Enrichment returned empty data, returning raw scraped data")
                    return final_df, output_csv
                    
            except Exception as e:
                print(f"\n⚠️ Enrichment failed: {e}")
                print("   Returning raw scraped data instead")
                import traceback
                traceback.print_exc()
                return final_df, output_csv

        except Exception as e:
            print(f"\n⚠️ Could not read final CSV for statistics: {e}")
            return final_df, output_csv

    except Exception as e:
        print(f"\n❌ PROCESS LEADS FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None, None
        

    finally:
        cleanup_shared_driver()