"""
============================================
FOUNDER EXTRACTOR MODULE - FIXED COLUMN MAPPING
============================================
✅ FIXED: Column names now match your CSV structure
✅ Maps: Founder_Name → Founder Name (code's internal name)
✅ Maps: Important_Person → Important Person
✅ Maps: LinkedIn_URL → LinkedIn Profile URL
✅ Maps: Industry_Sector → Industry
✅ Preserves existing data from your CSV columns

Your CSV Columns:
- Company_Name
- Funding_Amount
- Country
- Founder_Name (will be read and updated)
- Important_Person (will be read and updated)
- Email (will be read and updated)
- LinkedIn_URL (will be read and updated)
- Industry_Sector (will be read and updated)
- Category (will be read and updated)
"""

import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from mistralai import Mistral
import pandas as pd
import time
import re
from functools import wraps
import random
from itertools import cycle
import requests
import json

# ============================================
# COLUMN MAPPING CONFIGURATION
# ============================================

# Map your CSV column names to the code's internal column names
COLUMN_MAPPING = {
    'Company_Name': 'Company_Name',  # Company name column
    'Founder_Name': 'Founder Name',  # Your CSV → Code's internal name
    'Important_Person': 'Important Person',
    'Email': 'Email',
    'LinkedIn_URL': 'LinkedIn Profile URL',
    'Industry_Sector': 'Industry',
    'Category': 'Category'
}
# ============================================
# LLM CONFIGURATIONS (from main.py)
# ============================================

LLM_CONFIGS = {
    "Mistral": {
        "url": "https://api.mistral.ai/v1/chat/completions",
        "model": "mistral-small-latest",
        "auth_header": "Bearer"
    },
    "Claude": {
        "url": "https://api.anthropic.com/v1/messages",
        "model": "claude-3-5-sonnet-20241022",
        "auth_header": "x-api-key",
        "anthropic_version": "2023-06-01"
    },
    "Openai": {
        "url": "https://api.openai.com/v1/chat/completions",
        "model": "gpt-4o",
        "auth_header": "Bearer"
    },
    "Gemini": {
        "url": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        "model": "gemini-2.0-flash",
        "auth_header": "Bearer"
    },
    "Deepseek": {
        "url": "https://api.deepseek.com/chat/completions",
        "model": "deepseek-chat",
        "auth_header": "Bearer"
    },
    "Qwen": {
        "url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions",
        "model": "qwen-plus",
        "auth_header": "Bearer"
    },
    "Perplexity": {
        "url": "https://api.perplexity.ai/chat/completions",
        "model": "sonar",
        "auth_header": "Bearer"
    },
    "Llama": {
        "url": "https://api.llama-api.com/chat/completions",
        "model": "llama-3.3-70b",
        "auth_header": "Bearer"
    }
}
# ============================================
# GLOBAL CONFIGURATION VARIABLES
# ============================================

api_key_cycle = None
model_cycle = None
current_model = None
LLM_API_KEYS = []  # Changed from MISTRAL_API_KEYS
CURRENT_LLM_PROVIDER = "Mistral"  # NEW
CURRENT_LLM_MODEL = None  # NEW
APOLLO_API_KEY = ""

MAX_SEARCH_RETRIES = 3
MAX_MISTRAL_RETRIES = 3
RETRY_DELAY = 2

# Reverse mapping for saving back to CSV format
REVERSE_COLUMN_MAPPING = {v: k for k, v in COLUMN_MAPPING.items()}

# ============================================
# GLOBAL CONFIGURATION VARIABLES
# ============================================

api_key_cycle = None
model_cycle = None
current_model = None
MISTRAL_API_KEYS = []

MISTRAL_MODELS = ["mistral-small", "mistral-medium", "mistral-small-latest"]
APOLLO_API_KEY = ""

MAX_SEARCH_RETRIES = 3
MAX_MISTRAL_RETRIES = 3
RETRY_DELAY = 2


# ============================================
# INITIALIZATION FUNCTIONS
# ============================================
def initialize_cycles(api_keys, llm_provider="Mistral"):
    """Initialize the API key and model cycles with provided keys"""
    global api_key_cycle, model_cycle, current_model, LLM_API_KEYS, CURRENT_LLM_PROVIDER, CURRENT_LLM_MODEL
    
    # Match provider case-insensitively
    matched_provider = None
    for key in LLM_CONFIGS.keys():
        if key.lower() == llm_provider.lower():
            matched_provider = key
            break
    
    if matched_provider is None:
        print(f"❌ Error: {llm_provider} not supported")
        print(f"Available LLMs: {', '.join(LLM_CONFIGS.keys())}")
        raise ValueError(f"Unsupported LLM provider: {llm_provider}")
    
    CURRENT_LLM_PROVIDER = matched_provider
    LLM_API_KEYS = api_keys
    api_key_cycle = cycle(api_keys)
    
    # Set model based on provider
    CURRENT_LLM_MODEL = LLM_CONFIGS[matched_provider]["model"]
    current_model = CURRENT_LLM_MODEL
    
    print(f"✅ Initialized with {len(api_keys)} API keys for {matched_provider}")
    print(f"   Using model: {CURRENT_LLM_MODEL}")

def get_llm_client():
    """Returns API key and config for current LLM provider"""
    key = next(api_key_cycle)
    print(f"🔑 Using {CURRENT_LLM_PROVIDER} key: {key[:6]}...{key[-4:]}")
    return key, LLM_CONFIGS[CURRENT_LLM_PROVIDER]

def get_next_mistral_client():
    """Returns a new Mistral client using the next API key in rotation."""
    key = next(api_key_cycle)
    print(f"🔑 Switched to Mistral key: {key[:6]}...{key[-4:]}")
    return Mistral(api_key=key)


def rotate_model():
    """Rotates to the next model in the cycle"""
    global current_model
    current_model = next(model_cycle)
    print(f"🔄 Switched to model: {current_model}")
    return current_model


# ============================================
# COLUMN STANDARDIZATION FUNCTIONS
# ============================================

def standardize_columns(df):
    """
    Rename CSV columns to match code's internal column names
    
    Before: Founder_Name, Important_Person, LinkedIn_URL, Industry_Sector
    After: Founder Name, Important Person, LinkedIn Profile URL, Industry
    """
    print("\n🔧 STANDARDIZING COLUMN NAMES...")
    print("=" * 50)
    
    rename_map = {}
    for csv_col, internal_col in COLUMN_MAPPING.items():
        if csv_col in df.columns and csv_col != internal_col:
            rename_map[csv_col] = internal_col
            print(f"  📝 {csv_col} → {internal_col}")
    
    if rename_map:
        df = df.rename(columns=rename_map)
        print(f"✅ Renamed {len(rename_map)} columns")
    else:
        print("✅ No column renaming needed")
    
    return df


def restore_original_columns(df):
    """
    Restore original CSV column names before saving
    
    Before: Founder Name, Important Person, LinkedIn Profile URL, Industry
    After: Founder_Name, Important_Person, LinkedIn_URL, Industry_Sector
    """
    print("\n🔧 RESTORING ORIGINAL COLUMN NAMES...")
    
    rename_map = {}
    for internal_col, csv_col in REVERSE_COLUMN_MAPPING.items():
        if internal_col in df.columns and internal_col != csv_col:
            rename_map[internal_col] = csv_col
            print(f"  📝 {internal_col} → {csv_col}")
    
    if rename_map:
        df = df.rename(columns=rename_map)
        print(f"✅ Restored {len(rename_map)} column names")
    
    return df


# ============================================
# EXISTING DATA VALIDATION FUNCTIONS
# ============================================

def is_valid_data(value, field_type):
    """Check if existing data is valid and doesn't need research"""
    if pd.isna(value) or value is None:
        return False

    value_str = str(value).strip()

    if not value_str or value_str.lower() in [
        'data unavailable', 'not found', 'n/a', 'na', 'none', 'unknown', 
        'profile not found', '', 'null', 'nil', '-', '--', '---'
    ]:
        return False

    if field_type == 'founder' or field_type == 'ceo':
        if len(value_str.split()) < 2:
            return False
        if value_str.lower() in ['john doe', 'jane doe', 'founder name', 'ceo name']:
            return False
        return True

    elif field_type == 'linkedin':
        if 'linkedin.com' in value_str.lower() and '/in/' in value_str:
            return True
        return False

    elif field_type == 'email':
        if '@' in value_str and '.' in value_str:
            return True
        return False

    elif field_type == 'description':
        if len(value_str) > 10 and not value_str.lower().startswith('description'):
            return True
        return False

    elif field_type == 'category':
        valid_categories = ['b2b', 'b2c', 'b2b2c', 'saas', 'marketplace', 'enterprise', 'consumer', 'mixed']
        if any(cat in value_str.lower() for cat in valid_categories):
            return True
        return False

    elif field_type == 'industry':
        if len(value_str) > 3:
            return True
        return False

    return False


def analyze_existing_data(df, company_column='Company_Name'):
    """Analyze the input DataFrame to identify which fields need research"""
    print("\n🔍 ANALYZING EXISTING DATA...")
    print("=" * 50)

    total_rows = len(df)
    
    # Handle empty DataFrame
    if total_rows == 0:
        print("  ⚠️ DataFrame is empty (0 rows)")
        print("  ⚠️ No data to analyze - skipping analysis")
        print("=" * 50)
        return {}
    
    # Updated field mapping to match standardized column names
    field_mapping = {
        'Founder Name': 'founder',
        'Important Person': 'ceo', 
        'LinkedIn Profile URL': 'linkedin',
        'Email': 'email',
        'Description': 'description',
        'Category': 'category',
        'Industry': 'industry'
    }

    analysis = {}

    for col_name, field_type in field_mapping.items():
        if col_name in df.columns:
            valid_count = sum(1 for val in df[col_name] if is_valid_data(val, field_type))
            missing_count = total_rows - valid_count

            analysis[col_name] = {
                'valid': valid_count,
                'missing': missing_count,
                'percentage_complete': (valid_count / total_rows * 100)
            }

            print(f"  {col_name}:")
            print(f"    ✅ Valid: {valid_count} ({valid_count/total_rows*100:.1f}%)")
            print(f"    ❌ Missing: {missing_count} ({missing_count/total_rows*100:.1f}%)")
        else:
            analysis[col_name] = {
                'valid': 0,
                'missing': total_rows,
                'percentage_complete': 0
            }
            print(f"  {col_name}: ❌ Column not found - will create")

    total_missing_fields = sum(item['missing'] for item in analysis.values())
    print(f"\n📊 SUMMARY:")
    print(f"  Total companies: {total_rows}")
    print(f"  Total missing fields: {total_missing_fields}")
    print(f"  Research optimization: {100 - (total_missing_fields/(total_rows*len(field_mapping))*100):.1f}% of research can be skipped")

    return analysis

def should_research_field(row, field_name, field_type):
    """Determine if a specific field needs research for a given row"""
    if field_name not in row.index:
        return True

    return not is_valid_data(row[field_name], field_type)


# ============================================
# DECORATOR: Retry Logic with Model Rotation
# ============================================
def retry_on_failure(max_retries=3, delay=2, exceptions=(Exception,)):
    """Decorator to retry a function on failure with API key rotation."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            last_exception = None
            api_key_rotations = 0

            while True:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    retries += 1
                    last_exception = e
                    err = str(e).lower()

                    # Handle capacity exceeded errors
                    if "service_tier_capacity_exceeded" in err or "capacity" in err:
                        wait_time = 7
                        print(f"  ⚠️ {CURRENT_LLM_PROVIDER} at capacity (attempt {retries}/{max_retries}). Waiting {wait_time}s...")
                        time.sleep(wait_time)

                        # Rotate API key
                        try:
                            new_key, new_config = get_llm_client()
                            if 'api_key' in kwargs:
                                kwargs['api_key'] = new_key
                                kwargs['llm_config'] = new_config
                                api_key_rotations += 1
                                print(f"  🔄 Rotated to key: {new_key[:10]}...{new_key[-4:]}")

                            if api_key_rotations >= len(LLM_API_KEYS):
                                if retries >= max_retries:
                                    print(f"  ❌ Max retries reached after rotating all keys")
                                    return None
                                api_key_rotations = 0
                        except:
                            pass
                        
                        continue

                    # Handle rate limit errors
                    elif "429" in err or "rate" in err or "quota" in err or "limit" in err:
                        wait_time = 7
                        print(f"  ⚠️ {CURRENT_LLM_PROVIDER} rate limited (attempt {retries}/{max_retries}). Waiting {wait_time}s...")
                        time.sleep(wait_time)

                        # Rotate API key
                        try:
                            new_key, new_config = get_llm_client()
                            if 'api_key' in kwargs:
                                kwargs['api_key'] = new_key
                                kwargs['llm_config'] = new_config
                                api_key_rotations += 1
                                print(f"  🔄 Rotated to key: {new_key[:10]}...{new_key[-4:]}")

                            if api_key_rotations >= len(LLM_API_KEYS):
                                if retries >= max_retries:
                                    print(f"  ❌ Max retries reached after rotating all keys")
                                    return None
                                api_key_rotations = 0
                        except:
                            pass
                        
                        continue

                    # ✅ NEW: Handle 401 Unauthorized errors
                    elif "401" in err or "unauthorized" in err:
                        if retries >= max_retries:
                            print(f"  ❌ 401 Unauthorized after {max_retries} retries - API key is invalid")
                            return None
                        
                        wait_time = 3
                        print(f"  ⚠️ 401 Unauthorized (attempt {retries}/{max_retries}). Waiting {wait_time}s...")
                        time.sleep(wait_time)
                        
                        # Try to rotate to next API key
                        try:
                            new_key, new_config = get_llm_client()
                            if 'api_key' in kwargs:
                                kwargs['api_key'] = new_key
                                kwargs['llm_config'] = new_config
                                api_key_rotations += 1
                                print(f"  🔄 Rotated to key: {new_key[:10]}...{new_key[-4:]}")
                            
                            if api_key_rotations >= len(LLM_API_KEYS):
                                print(f"  ❌ All {len(LLM_API_KEYS)} API keys failed with 401")
                                return None
                        except Exception as rotate_err:
                            print(f"  ⚠️ Could not rotate key: {str(rotate_err)[:50]}")
                        
                        continue

                    # Handle all other errors
                    else:
                        if retries >= max_retries:
                            print(f"  ❌ Max retries reached for {func.__name__}: {str(e)[:100]}")
                            return None
                        
                        print(f"  ⚠️ {func.__name__} failed ({retries}/{max_retries}): {str(e)[:100]}...")
                        time.sleep(delay)
                        continue

            # Should never reach here, but just in case
            return None
            
        return wrapper
    return decorator


# ============================================
# [REST OF THE CODE REMAINS THE SAME]
# ============================================

def search_google_with_fallback(company_name, search_type, driver):
    """Simplified version — always uses full Google search page text"""
    if search_type == 'founder':
        search_queries = [
            f"{company_name} founder name",
            f"{company_name} co-founder",
            f"who founded {company_name}",
        ]
    else:
        search_queries = [
            f"{company_name} CEO name",
            f"{company_name} key people",
            f"who is the CEO of {company_name}",
        ]

    best_result = ""
    best_query = ""

    for query in search_queries:
        try:
            search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
            driver.get(search_url)
            time.sleep(4)

            page_text = driver.find_element(By.TAG_NAME, 'body').text
            result_text = page_text[:2000]

            if len(result_text) > len(best_result):
                best_result = result_text
                best_query = query

            if len(result_text) > 1000:
                print(f"  ✅ Found good result with query: '{query}'")
                break

        except Exception as e:
            print(f"  ⚠️ Query '{query}' failed: {str(e)[:80]}")
            continue

    if not best_result:
        raise Exception(f"All search strategies failed for {company_name}")

    print(f"  📄 Using {len(best_result)} characters from '{best_query}' search")
    return best_result


@retry_on_failure(max_retries=MAX_SEARCH_RETRIES, delay=RETRY_DELAY)
def find_linkedin_profile(person_name, company_name, driver):
    """Searches Google for the LinkedIn profile of a person."""
    if not person_name or person_name == "Data unavailable" or person_name == "UNABLE_TO_DECIDE":
        print(f"  ⚠️ No valid person name to search LinkedIn for")
        return "Profile not found"

    if company_name.lower().replace("private", "").replace("limited", "").strip() in person_name.lower():
        print(f"  ⚠️ Person name looks like company name, skipping LinkedIn search")
        return "Profile not found"

    query = f"{person_name} {company_name} LinkedIn profile"
    search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}"

    print(f"  🔍 Searching LinkedIn for: {person_name} ({company_name})")
    driver.get(search_url)
    time.sleep(4)

    try:
        page_source = driver.page_source
        linkedin_urls = re.findall(r'https://[a-z]{0,3}\.?linkedin\.com/[^\s"\'<>]+', page_source)
        linkedin_profiles = [url.split('?')[0] for url in linkedin_urls if '/in/' in url or '/company/' in url]
        linkedin_profiles = list(dict.fromkeys(linkedin_profiles))

        if linkedin_profiles:
            print(f"  ✅ Found LinkedIn URL: {linkedin_profiles[0]}")
            return linkedin_profiles[0]
        else:
            print("  ⚠️ No LinkedIn profile found on page.")
            return "Profile not found"
    except Exception as e:
        print(f"  ❌ LinkedIn search error: {str(e)[:100]}")
        return "Profile not found"


def extract_linkedin_id(linkedin_url: str) -> str:
    """Extract LinkedIn profile ID/username from URL"""
    if not linkedin_url or linkedin_url == "Profile not found":
        return None

    try:
        parts = linkedin_url.rstrip('/').split('/')
        if 'in' in parts:
            idx = parts.index('in')
            if idx + 1 < len(parts):
                return parts[idx + 1].split('?')[0]
        return None
    except Exception as e:
        print(f"    ⚠️ Error extracting LinkedIn ID: {str(e)[:50]}")
        return None


def fetch_contact_info_apollo(person_name, company_name, linkedin_url=None, apollo_api_key=None):
    """Fetch email and phone using Apollo People Match endpoint"""

    if not apollo_api_key:
        apollo_api_key = APOLLO_API_KEY

    if not apollo_api_key:
        print("  ⚠️ No Apollo API key configured")
        return "", ""

    if not linkedin_url or linkedin_url == "Profile not found":
        print(f"  ⚠️ No LinkedIn URL available for Apollo lookup")
        return "", ""

    if not person_name or person_name == "Data unavailable":
        print(f"  ⚠️ Invalid person name for Apollo lookup")
        return "", ""

    linkedin_id = extract_linkedin_id(linkedin_url)
    if not linkedin_id:
        print(f"  ⚠️ Could not extract LinkedIn ID from URL")
        return "", ""

    print(f"  🔍 Apollo API: Looking up {person_name} via LinkedIn")

    url = "https://api.apollo.io/v1/people/match"

    headers = {
        "Content-Type": "application/json",
        "Cache-Control": "no-cache",
        "X-Api-Key": apollo_api_key
    }

    payload = {
        "api_key": apollo_api_key,
        "linkedin_url": linkedin_url
    }

    try:
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=10
        )

        if response.status_code == 200:
            data = response.json()
            person = data.get('person', {})

            email = person.get('email', '')
            email_status = person.get('email_status', '')

            phone_numbers = person.get('phone_numbers', [])
            phone = phone_numbers[0].get('sanitized_number', '') if phone_numbers else ''

            if not phone:
                phone = person.get('sanitized_phone', '') or person.get('phone', '')

            if email:
                status_icon = "✓" if email_status == 'verified' else "~"
                email_display = email[:30] + '...' if len(email) > 30 else email
                print(f"    {status_icon} Found email: {email_display} ({email_status})")
            else:
                print(f"    ✗ No email found in Apollo")

            if phone:
                print(f"    ✓ Found phone: {phone}")

            return email, phone

        elif response.status_code == 404:
            print(f"    ✗ Profile not found in Apollo database")
            return "", ""

        elif response.status_code == 429:
            print(f"    ⚠️ Apollo rate limit hit, waiting 5s...")
            time.sleep(5)
            return fetch_contact_info_apollo(person_name, company_name, linkedin_url, apollo_api_key)

        elif response.status_code == 403:
            print(f"    ❌ Apollo 403 Forbidden - Check API key")
            return "", ""

        elif response.status_code == 401:
            print(f"    ❌ Apollo: Invalid API key")
            return "", ""

        else:
            print(f"    ✗ Apollo API error: {response.status_code}")
            return "", ""

    except requests.exceptions.Timeout:
        print(f"    ✗ Apollo API timeout")
        return "", ""

    except requests.exceptions.RequestException as e:
        print(f"    ✗ Apollo request failed: {str(e)[:100]}")
        return "", ""

    except json.JSONDecodeError:
        print(f"    ✗ Apollo returned invalid JSON")
        return "", ""

    except Exception as e:
        print(f"    ✗ Apollo unexpected error: {str(e)[:100]}")
        return "", ""


@retry_on_failure(max_retries=MAX_MISTRAL_RETRIES, delay=RETRY_DELAY)
def extract_with_llm(company_name, google_response, extraction_type, api_key, llm_config):
    """Uses configured LLM to extract a clean founder or CEO name"""
    global CURRENT_LLM_MODEL, CURRENT_LLM_PROVIDER

    if extraction_type == 'founder':
        role = "founder"
        label = "Founder name(s):"
    else:
        role = "CEO or key executive"
        label = "Key Person name:"

    prompt = f"""
You are an information extraction system.
Extract ONLY the {role}'s full name(s) from the given text.

Company: "{company_name}"
Context:
{google_response}

Rules:
- Output ONLY the name(s), separated by commas if multiple.
- If the designation (e.g., CEO, Founder) is mentioned, include it in parentheses (e.g., John Doe (CEO)).
- If you cannot find any valid person name, output exactly: UNABLE_TO_EXTRACT
- Do NOT include explanations, reasoning, or commentary.
- Your entire response must be ONE line with only the name(s) or UNABLE_TO_EXTRACT.

{label}
"""

    print(f"  🤖 Calling {CURRENT_LLM_PROVIDER} ({CURRENT_LLM_MODEL}) for {company_name} ({extraction_type})...")

    # Build request based on LLM type
    if CURRENT_LLM_PROVIDER.lower() == "claude":
        headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": llm_config["anthropic_version"]
        }
        data = {
            "model": llm_config["model"],
            "max_tokens": 100,
            "messages": [{"role": "user", "content": prompt}]
        }
    else:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"{llm_config['auth_header']} {api_key}"
        }
        data = {
            "model": llm_config["model"],
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 100,
            "temperature": 0.0
        }

    try:
        response = requests.post(llm_config["url"], headers=headers, json=data, timeout=30)
        response.raise_for_status()
        result = response.json()
        
        # Extract content based on provider
        if CURRENT_LLM_PROVIDER.lower() == "claude":
            extracted = result['content'][0]['text'].strip()
        else:
            extracted = result['choices'][0]['message']['content'].strip()

    except requests.exceptions.RequestException as e:
        print(f"  ❌ {CURRENT_LLM_PROVIDER} API error: {str(e)}")
        raise

    # [Keep rest of your existing cleaning logic]
    forbidden_phrases = [
        "based on the provided information",
        "no names are directly mentioned",
        "cannot provide",
        "no name found",
    ]

    for phrase in forbidden_phrases:
        extracted = re.sub(re.escape(phrase), "", extracted, flags=re.IGNORECASE)

    extracted = extracted.strip()
    if not extracted or len(extracted) < 2 or extracted.lower() in [
        "unable_to_extract", "not found", "none", "n/a", "unknown",
    ]:
        return "Data unavailable"

    return extracted

@retry_on_failure(max_retries=MAX_MISTRAL_RETRIES, delay=RETRY_DELAY)
def extract_company_info(company_name, api_key, llm_config):
    """Extracts company description, category, and industry"""
    global CURRENT_LLM_MODEL, CURRENT_LLM_PROVIDER

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
    
    # Build request based on LLM type
    if CURRENT_LLM_PROVIDER.lower() == "claude":
        headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": llm_config["anthropic_version"]
        }
        data = {
            "model": llm_config["model"],
            "max_tokens": 120,
            "messages": [{"role": "user", "content": prompt}]
        }
    else:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"{llm_config['auth_header']} {api_key}"
        }
        data = {
            "model": llm_config["model"],
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 120,
            "temperature": 0.2
        }
    
    try:
        response = requests.post(llm_config["url"], headers=headers, json=data, timeout=30)
        response.raise_for_status()
        result = response.json()
        
        # Extract content based on provider
        if CURRENT_LLM_PROVIDER.lower() == "claude":
            output = result['content'][0]['text'].strip()
        else:
            output = result['choices'][0]['message']['content'].strip()
        
        description = "Data unavailable"
        category = "Data unavailable"
        industry = "Data unavailable"

        desc_match = re.search(r"Description:\s*(.+)", output)
        cat_match = re.search(r"Category:\s*(.+)", output)
        ind_match = re.search(r"Industry:\s*(.+)", output)

        if desc_match:
            description = desc_match.group(1).strip().split("\n")[0][:200]
        if cat_match:
            category = cat_match.group(1).strip().split("\n")[0]
        if ind_match:
            industry = ind_match.group(1).strip().split("\n")[0]

        return description, category, industry

    except Exception as e:
        print(f"  ⚠️ Company info extraction failed: {str(e)}")
        return "Data unavailable", "Data unavailable", "Data unavailable"
    
@retry_on_failure(max_retries=MAX_MISTRAL_RETRIES, delay=RETRY_DELAY)
def reconcile_founder_ceo(company_name, founder_response, ceo_response, api_key, llm_config):
    """Uses LLM to intelligently decide the true founder"""
    global CURRENT_LLM_MODEL, CURRENT_LLM_PROVIDER

    prompt = f"""
You are an expert in company data extraction. Your goal is to determine the *true, most relevant founder*
of the company based on both historical and current leadership information.

Company: "{company_name}"

Founder context:
{founder_response[:1500]}

CEO context:
{ceo_response[:1500]}

Rules:
- Return ONLY the person's name — never the company name.
- If you cannot find any person, return exactly "UNABLE_TO_DECIDE".
- Prefer a person who is *currently leading the company* if there was a rebrand or merger.
- If multiple people are mentioned, choose the one most strongly tied to this company.
- If the same person appears as both founder and CEO, pick that one.
- Do NOT include explanations, extra words, or reasoning.
- Output must be exactly one line with the person's full name (e.g., "John Doe").

Final Founder:
"""

    # Build request based on LLM type
    if CURRENT_LLM_PROVIDER.lower() == "claude":
        headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": llm_config["anthropic_version"]
        }
        data = {
            "model": llm_config["model"],
            "max_tokens": 50,
            "messages": [{"role": "user", "content": prompt}]
        }
    else:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"{llm_config['auth_header']} {api_key}"
        }
        data = {
            "model": llm_config["model"],
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 50,
            "temperature": 0.1
        }

    try:
        response = requests.post(llm_config["url"], headers=headers, json=data, timeout=30)
        response.raise_for_status()
        result = response.json()
        
        # Extract content based on provider
        if CURRENT_LLM_PROVIDER.lower() == "claude":
            raw_output = result['content'][0]['text'].strip()
        else:
            raw_output = result['choices'][0]['message']['content'].strip()

    except Exception as e:
        print(f"  ⚠️ Reconciliation failed: {str(e)}")
        return "Data unavailable"

    cleaned = re.sub(r'(?i)based on.*?(is|are)\s+', '', raw_output)
    cleaned = re.sub(r'\*|\(|\)|\[|\]', '', cleaned)
    cleaned = re.sub(r'[^A-Za-z\s\.-]', '', cleaned).strip()

    name_match = re.search(r'([A-Z][a-z]+\s+[A-Z][a-z]+)', cleaned)
    if name_match:
        cleaned = name_match.group(1)

    if company_name.lower().replace("private", "").replace("limited", "").strip() in cleaned.lower():
        print(f"  ⚠️ Reconciliation output looked like a company name; discarding.")
        return "Data unavailable"

    return cleaned

def cross_validate_names(founder_name, ceo_name):
    """If both founder and CEO names overlap, prefer the CEO name"""
    if not founder_name or not ceo_name:
        return founder_name
    if founder_name.lower() in ceo_name.lower() or ceo_name.lower() in founder_name.lower():
        return ceo_name
    return founder_name

def process_single_company(company_name, row, driver):
    """Process a single company to extract founder and CEO"""
    result = {
        'founder_name': 'Data unavailable',
        'founder_response': 'Failed to retrieve',
        'ceo_name': 'Data unavailable',
        'ceo_response': 'Failed to retrieve',
        'linkedin_url': 'Profile not found',
        'email': '',
        'phone': '',
        'description': 'Data unavailable',
        'category': 'Data unavailable',
        'industry': 'Data unavailable'
    }

    print(f"  📋 Checking existing data...")

    need_founder_research = should_research_field(row, 'Founder Name', 'founder')
    if not need_founder_research:
        result['founder_name'] = row['Founder Name']
        print(f"  ✅ Using existing founder: {result['founder_name']}")

    need_ceo_research = should_research_field(row, 'Important Person', 'ceo')
    if not need_ceo_research:
        result['ceo_name'] = row['Important Person']
        print(f"  ✅ Using existing CEO: {result['ceo_name']}")

    need_linkedin_research = should_research_field(row, 'LinkedIn Profile URL', 'linkedin')
    if not need_linkedin_research:
        result['linkedin_url'] = row['LinkedIn Profile URL']
        print(f"  ✅ Using existing LinkedIn: {result['linkedin_url']}")

    need_email_research = should_research_field(row, 'Email', 'email')
    if not need_email_research:
        result['email'] = row['Email']
        print(f"  ✅ Using existing email: {result['email']}")

    need_description_research = should_research_field(row, 'Description', 'description')
    need_category_research = should_research_field(row, 'Category', 'category')
    need_industry_research = should_research_field(row, 'Industry', 'industry')

    if not need_description_research:
        result['description'] = row['Description']
        print(f"  ✅ Using existing description: {result['description'][:50]}...")

    if not need_category_research:
        result['category'] = row['Category']
        print(f"  ✅ Using existing category: {result['category']}")

    if not need_industry_research:
        result['industry'] = row['Industry']
        print(f"  ✅ Using existing industry: {result['industry']}")

    if need_description_research or need_category_research or need_industry_research:
        print(f"  🧠 Fetching missing company info for {company_name}...")
        api_key, llm_config = get_llm_client()
        description, category, industry = extract_company_info(company_name, api_key, llm_config)

        if need_description_research:
            result['description'] = description
        if need_category_research:
            result['category'] = category
        if need_industry_research:
            result['industry'] = industry
    else:
        print(f"  ⏩ Skipping company info research - all data available")

    if need_founder_research:
        print(f"  🔍 Searching for founder...")
        try:
            founder_response = search_google_with_fallback(company_name, 'founder', driver)
            if founder_response:
                result['founder_response'] = founder_response[:1000]
                print(f"  📄 Founder response: {len(founder_response)} characters retrieved")

                print(f"  🤖 Extracting founder name with {CURRENT_LLM_PROVIDER}...")
                api_key, llm_config = get_llm_client()
                founder_name = extract_with_llm(company_name, founder_response, 'founder', api_key, llm_config)
                if founder_name:
                    result['founder_name'] = founder_name
                    print(f"  ✅ Founder: {founder_name}")
                else:
                    print(f"  ⚠️ Could not extract founder name")
        except Exception as e:
            print(f"  ❌ Founder extraction failed: {str(e)[:100]}")
    else:
        print(f"  ⏩ Skipping founder research - data available")

    time.sleep(2)

    if need_ceo_research:
        print(f"  🔍 Searching for CEO/Key Person...")
        try:
            ceo_response = search_google_with_fallback(company_name, 'ceo', driver)
            if ceo_response:
                result['ceo_response'] = ceo_response[:1000]
                print(f"  📄 CEO response: {len(ceo_response)} characters retrieved")

                print(f"  🤖 Extracting CEO/Key Person name with {CURRENT_LLM_PROVIDER}...")
                api_key, llm_config = get_llm_client()
                ceo_name = extract_with_llm(company_name, ceo_response, 'ceo', api_key, llm_config)
                if ceo_name:
                    result['ceo_name'] = ceo_name
                    print(f"  ✅ CEO/Key Person: {ceo_name}")
                else:
                    print(f"  ⚠️ Could not extract CEO name")
        except Exception as e:
            print(f"  ❌ CEO extraction failed: {str(e)[:100]}")
    else:
        print(f"  ⏩ Skipping CEO research - data available")

    if need_founder_research or need_ceo_research:
        smart_founder_decision = result['founder_name']

        try:
            result['founder_name'] = cross_validate_names(result['founder_name'], result['ceo_name'])

            if need_founder_research and need_ceo_research:
                print(f"  🧩 Reconciling founder and CEO data...")
               


                api_key, llm_config = get_llm_client()
                reconciled_founder = reconcile_founder_ceo(
    company_name,
    result['founder_response'],
    result['ceo_response'],
    api_key,
    llm_config
)
                if reconciled_founder and reconciled_founder != "UNABLE_TO_DECIDE" and reconciled_founder != "Data unavailable":
                    result['founder_name'] = reconciled_founder
                    smart_founder_decision = reconciled_founder
                    print(f"  ✅ Smart Founder Decision: {reconciled_founder}")
                else:
                    smart_founder_decision = result['founder_name']
        except Exception as e:
            print(f"  ⚠️ Reconciliation error: {str(e)}")
            smart_founder_decision = result['founder_name']
    else:
        smart_founder_decision = result['founder_name']

    if need_founder_research or need_ceo_research:
        try:
            founder_raw = result['founder_name']
            ceo_raw = result['ceo_name']

            names = [n.strip() for n in re.split(r',|&|and', founder_raw) if n.strip()]
            unique_names = []
            for n in names:
                if n.lower() not in ['founder', 'co-founder', 'ceo', 'director'] and len(n.split()) >= 2:
                    if n not in unique_names:
                        unique_names.append(n)

            primary_founder = ""
            extra_names = []

            if unique_names:
                for n in unique_names:
                    if "ceo" in n.lower():
                        primary_founder = n
                        break
                if not primary_founder:
                    for n in unique_names:
                        if "founder" in n.lower():
                            primary_founder = n
                            break
                if not primary_founder:
                    primary_founder = unique_names[0]

                extra_names = [n for n in unique_names if n != primary_founder]

            if primary_founder:
                result['founder_name'] = primary_founder.strip()
                smart_founder_decision = primary_founder.strip()
            if extra_names:
                ceo_combined = ceo_raw
                for n in extra_names:
                    if n not in ceo_combined:
                        ceo_combined += f", {n}"
                result['ceo_name'] = ceo_combined.strip().strip(',')

        except Exception as e:
            print(f"  ⚠️ Post-processing error: {str(e)}")

    def fix_unbalanced_parentheses(text):
        text = text.strip()
        if text.count('(') > text.count(')'):
            text += ')'
        elif text.count(')') > text.count('('):
            text = '(' + text
        return text

    result['founder_name'] = fix_unbalanced_parentheses(result['founder_name'])
    result['ceo_name'] = fix_unbalanced_parentheses(result['ceo_name'])
    smart_founder_decision = fix_unbalanced_parentheses(smart_founder_decision)

    if need_linkedin_research:
        linkedin_url = "Profile not found"

        try:
            if smart_founder_decision and smart_founder_decision != "Data unavailable":
                print(f"  🔗 Using smart founder decision for LinkedIn: {smart_founder_decision}")
                linkedin_url = find_linkedin_profile(smart_founder_decision, company_name, driver)

            if linkedin_url == "Profile not found" and result['ceo_name'] != "Data unavailable":
                ceo_first = result['ceo_name'].split(',')[0].strip()
                if ceo_first and ceo_first != "Data unavailable":
                    print(f"  🔗 Founder LinkedIn not found, trying CEO: {ceo_first}")
                    linkedin_url = find_linkedin_profile(ceo_first, company_name, driver)
        except Exception as e:
            print(f"  ⚠️ LinkedIn lookup failed: {str(e)}")

        result['linkedin_url'] = linkedin_url
    else:
        print(f"  ⏩ Skipping LinkedIn research - data available")

    if need_email_research:
        print(f"  📧 Fetching contact info from Apollo API...")
        email, phone = fetch_contact_info_apollo(
            smart_founder_decision, 
            company_name, 
            linkedin_url=result['linkedin_url'],
            apollo_api_key=APOLLO_API_KEY
        )
        result['email'] = email
    else:
        print(f"  ⏩ Skipping Apollo research - email available")

    return result


def setup_driver():
    """Setup and return undetected Chrome driver"""
    print("\n🚀 Starting Undetected Chrome browser...")

    options = uc.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-gpu')
    options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')

    driver = uc.Chrome(options=options, version_main=None)
    driver.maximize_window()

    return driver


def process_companies(df, company_column='Company_Name'):
    # mistral_client parameter removed - we get it from get_llm_client() now
    """Main function to process all companies"""

    # Standardize column names first
    df = standardize_columns(df)

    analysis = analyze_existing_data(df, company_column)

    required_columns = {
        'Founder Name': 'Data unavailable',
        'Important Person': 'Data unavailable', 
        'LinkedIn Profile URL': 'Profile not found',
        'Email': '',
        'Description': 'Data unavailable',
        'Category': 'Data unavailable',
        'Industry': 'Data unavailable'
    }

    for col, default_val in required_columns.items():
        if col not in df.columns:
            df[col] = default_val
            print(f"  ➕ Added missing column: {col}")

    founder_names = []
    founder_responses = []
    ceo_names = []
    ceo_responses = []
    linkedin_urls = []
    emails = []
    descriptions = []
    categories = []
    industries = []

    driver = setup_driver()

    try:
        total = len(df)
        research_count = 0
        skipped_count = 0

        for idx, row in df.iterrows():
            company_name = row[company_column]

            print(f"\n{'='*70}")
            print(f"[{idx+1}/{total}] Processing: {company_name}")
            print(f"{'='*70}")

            needs_research = (
                should_research_field(row, 'Founder Name', 'founder') or
                should_research_field(row, 'Important Person', 'ceo') or
                should_research_field(row, 'LinkedIn Profile URL', 'linkedin') or
                should_research_field(row, 'Email', 'email') or
                should_research_field(row, 'Description', 'description') or
                should_research_field(row, 'Category', 'category') or
                should_research_field(row, 'Industry', 'industry')
            )

            if not needs_research:
                print(f"  ✅ All data available - SKIPPING RESEARCH")
                result = {
                    'founder_name': row.get('Founder Name', 'Data unavailable'),
                    'founder_response': 'Existing data used',
                    'ceo_name': row.get('Important Person', 'Data unavailable'),
                    'ceo_response': 'Existing data used',
                    'linkedin_url': row.get('LinkedIn Profile URL', 'Profile not found'),
                    'email': row.get('Email', ''),
                    'description': row.get('Description', 'Data unavailable'),
                    'category': row.get('Category', 'Data unavailable'),
                    'industry': row.get('Industry', 'Data unavailable')
                }
                skipped_count += 1
            else:
                print(f"  🔍 Research needed - processing...")
                result = process_single_company(company_name, row, driver)
                research_count += 1

            founder_names.append(result['founder_name'])
            founder_responses.append(result['founder_response'])
            ceo_names.append(result['ceo_name'])
            ceo_responses.append(result['ceo_response'])
            linkedin_urls.append(result.get('linkedin_url', 'Profile not found'))
            emails.append(result.get('email', ''))
            descriptions.append(result.get('description', 'Data unavailable'))
            categories.append(result.get('category', 'Data unavailable'))
            industries.append(result.get('industry', 'Data unavailable'))

            if idx < total - 1:
                delay = 3 + (idx % 3)
                print(f"  ⏳ Waiting {delay} seconds before next company...")
                time.sleep(delay)

        print(f"\n📊 PROCESSING SUMMARY:")
        print(f"  🔍 Companies researched: {research_count}")
        print(f"  ⏩ Companies skipped (data complete): {skipped_count}")
        print(f"  📈 Research efficiency: {skipped_count/total*100:.1f}% of work avoided")

    except KeyboardInterrupt:
        print("\n\n⚠️ Process interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
    finally:
        driver.quit()
        print("\n✅ Browser closed")

    df['Founder Name'] = founder_names
    df['Important Person'] = ceo_names
    df['LinkedIn Profile URL'] = linkedin_urls
    df['Email'] = emails
    df['Description'] = descriptions
    df['Category'] = categories
    df['Industry'] = industries

    return df


def extract_company_data(
    csv_path,
    mistral_api_keys=None,  # Keep for backward compatibility
    output_path=None,
    company_column='Company_Name',
    apollo_api_key=None,
    enable_reprocessing=True,
    save_intermediate=True,
    llm_provider="Mistral",  # NEW PARAMETER
    llm_api_keys=None  # NEW PARAMETER - list of keys
):
    """
    Extract founder, CEO, and company information from a CSV file.
    ✅ FIXED: Column names now match your CSV structure
    """

    global APOLLO_API_KEY
    if apollo_api_key:
        APOLLO_API_KEY = apollo_api_key
    elif not APOLLO_API_KEY:
        APOLLO_API_KEY = ""

    # Handle backward compatibility
    if llm_api_keys:
     api_keys_to_use = llm_api_keys
    elif mistral_api_keys:
     api_keys_to_use = mistral_api_keys
    else:
     raise ValueError("Either llm_api_keys or mistral_api_keys must be provided")

    initialize_cycles(api_keys_to_use, llm_provider)

    print("=" * 70)
    print("COMPANY FOUNDER & CEO EXTRACTOR - FIXED COLUMN MAPPING")
    print("✅ Now matches your CSV column names")
    print("=" * 70)
    print(f"Start Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

  

    print(f"\n🔑 Initializing {llm_provider} LLM...")
    try:
      api_key, llm_config = get_llm_client()
      print(f"✅ {CURRENT_LLM_PROVIDER} initialized with model: {CURRENT_LLM_MODEL}")
    except Exception as e:
        print(f"❌ Failed to initialize Mistral: {e}")
        raise

    
    try:
     if isinstance(csv_path, pd.DataFrame):
        df = csv_path.copy()
        print(f"✅ Received DataFrame with {len(df)} rows")
     else:
        df = pd.read_csv(csv_path)
        print(f"✅ Successfully loaded CSV with {len(df)} rows")
     print(f"📋 Original columns: {', '.join(df.columns.tolist())}")
    except FileNotFoundError:
        print(f"❌ Error: CSV file not found at: {csv_path}")
        raise
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        raise

    if company_column not in df.columns:
        available_cols = ', '.join(df.columns.tolist())
        raise ValueError(
            f"❌ Column '{company_column}' not found in CSV.\n"
            f"Available columns: {available_cols}"
        )

    result_df = process_companies(df, company_column=company_column)

    # Restore original column names before saving
    result_df = restore_original_columns(result_df)

    if output_path:
        print(f"\n💾 Saving final results to: {output_path}")
        try:
            result_df.to_excel(output_path, index=False, engine='openpyxl')
            print(f"✅ Final results saved successfully!")
            print(f"📋 Saved columns: {', '.join(result_df.columns.tolist())}")
        except Exception as e:
            print(f"❌ Error saving results: {e}")

    print("\n" + "=" * 70)
    print("PROCESSING COMPLETE")
    print("=" * 70)
    print(f"End Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    return result_df, output_path