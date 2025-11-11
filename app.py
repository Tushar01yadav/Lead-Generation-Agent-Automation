import streamlit as st
import requests
import json
import pandas as pd
import os
import subprocess
import threading
import random
import re
from ipo import scrape_zerodha_ipos
import time
from datetime import datetime, timedelta
import base64
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
import pickle
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shutil

# 👇 Check if Chrome or Chromium is installed in the environment
chrome_path = shutil.which("google-chrome") or shutil.which("chromium-browser")
st.write("🧠 Chrome path detected:", chrome_path or "❌ No Chrome found")

# You can also log it to console for debugging
print("Chrome path:", chrome_path or "No Chrome found")


os.environ['STREAMLIT_DEBUG'] = '1'
# Replace the hardcoded TRACKING_SERVER_URL line with:

# Dynamic tracking server URL (updated when server starts)
if 'tracking_server_url' in st.session_state and st.session_state.tracking_server_url:
    TRACKING_SERVER_URL = st.session_state.tracking_server_url
else:
    TRACKING_SERVER_URL = "https://your-default-url.trycloudflare.com"  # Fallback
# ============ EMAIL TEMPLATES ============
EMAIL_TEMPLATES = {
    "healthtech": """Subject: Revolutionizing Healthcare with Innovation

Dear {name},

I hope this email finds you well. I came across your company {company_name} and was impressed by your work in the healthcare technology sector.

We specialize in connecting innovative healthtech startups with potential investors and partners. I believe there could be valuable opportunities for collaboration.

Would you be interested in a brief 15-minute call to explore potential synergies?

Best regards,
{sender_name}""",
    
    "edtech": """Subject: Transforming Education Through Technology

Dear {name},

I recently discovered {company_name} and was inspired by your mission to transform education through technology.

We work with leading edtech companies to accelerate their growth and expand their market reach. I'd love to discuss how we might support your vision.

Are you available for a quick conversation next week?

Warm regards,
{sender_name}""",
    
    "real_estate": """Subject: Innovation in Real Estate & PropTech

Dear {name},

Your work at {company_name} in the real estate technology space caught my attention. The proptech industry is evolving rapidly, and innovative companies like yours are leading the way.

I'd appreciate the opportunity to learn more about your vision and explore potential collaboration opportunities.

Would you be open to a brief call?

Best regards,
{sender_name}""",
    
    "ai": """Subject: AI Innovation & Growth Opportunities

Dear {name},

I've been following the impressive work {company_name} is doing in artificial intelligence. Your approach to {specific_area} is particularly innovative.

We connect AI startups with strategic partners and growth opportunities. I believe there could be meaningful ways we could collaborate.

Could we schedule a short call to discuss further?

Kind regards,
{sender_name}""",
    
    "customize": """Subject: 

Dear {name},

Best regards,
{sender_name}"""
}

def start_tracking_server():
    """
    Start tracking server and cloudflared tunnel in background
    Returns the tunnel URL
    """
    try:
        # Check if cloudflared is installed
        try:
            subprocess.run(["cloudflared", "--version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False, "❌ Cloudflared is not installed. Please install it first: https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/installation/"
        
        # Check if port 5001 is already in use
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('localhost', 5001))
        sock.close()
        
        if result == 0:
            log_to_debug("⚠️ Port 5001 already in use - server may already be running")
        else:
            # Start Flask tracking server in background
            log_to_debug("🚀 Starting Flask tracking server on port 5001...")
            flask_process = subprocess.Popen(
                ["python", "tracking_server.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            st.session_state.flask_process = flask_process
            time.sleep(2)  # Give Flask time to start
        
        # Start cloudflared tunnel
        log_to_debug("🌐 Creating Cloudflare tunnel...")
        tunnel_process = subprocess.Popen(
            ["cloudflared", "tunnel", "--url", "http://localhost:5001"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        st.session_state.tunnel_process = tunnel_process
        
        # Extract tunnel URL from output
        tunnel_url = None
        max_wait = 15  # Wait up to 15 seconds
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            line = tunnel_process.stderr.readline()
            if line:
                log_to_debug(f"Cloudflare: {line.strip()}")
                # Look for the URL pattern
                match = re.search(r'https://[a-z0-9-]+\.trycloudflare\.com', line)
                if match:
                    tunnel_url = match.group(0)
                    break
            time.sleep(0.5)
        
        if tunnel_url:
            log_to_debug(f"✅ Tunnel URL extracted: {tunnel_url}")
            return True, tunnel_url
        else:
            log_to_debug("❌ Failed to extract tunnel URL")
            return False, "Failed to get tunnel URL. Check if cloudflared is running correctly."
    
    except Exception as e:
        log_to_debug(f"❌ Error starting tracking server: {str(e)}")
        return False, str(e)

def stop_tracking_server():
    """Stop tracking server and tunnel"""
    try:
        if 'tunnel_process' in st.session_state and st.session_state.tunnel_process:
            st.session_state.tunnel_process.terminate()
            log_to_debug("🛑 Cloudflare tunnel stopped")
        
        if 'flask_process' in st.session_state and st.session_state.flask_process:
            st.session_state.flask_process.terminate()
            log_to_debug("🛑 Flask server stopped")
        
        if 'tracking_server_url' in st.session_state:
            del st.session_state.tracking_server_url
        
        return True
    except Exception as e:
        log_to_debug(f"❌ Error stopping server: {str(e)}")
        return False

# Add this helper function
def debug_llm_config(config, label="LLM Config"):
    """Debug helper to inspect LLM config"""
    print(f"\n🔍 {label}:")
    if isinstance(config, dict):
        for key, value in config.items():
            if 'key' in key.lower():
                print(f"  • {key}: {value[:10] if value else 'EMPTY'}...")
            else:
                print(f"  • {key}: {value}")
    else:
        print(f"  • Type: {type(config)}")
        print(f"  • Value: {config}")

# Page config with custom sidebar width
st.set_page_config(
    page_title="Lead Generation Agent", 
    page_icon="📊", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for sidebar width
st.markdown("""
<style>
    [data-testid="stSidebar"] {
        width: 19% !important;
    }
</style>
""", unsafe_allow_html=True)

# ============ UTILITY FUNCTIONS ============

def log_to_debug(message):
    """Add message to debug console"""
    if "debug_logs" not in st.session_state:
        st.session_state.debug_logs = []
    st.session_state.debug_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")

def get_default_endpoint(provider):
    """Get default API endpoint for provider"""
    endpoints = {
        "OpenAI": "https://api.openai.com/v1/chat/completions",
        "Claude": "https://api.anthropic.com/v1/messages",
        "Gemini": "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent",
        "Mistral": "https://api.mistral.ai/v1/chat/completions",
        "Deepseek": "https://api.deepseek.com/v1/chat/completions",
        "Qwen": "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation",
        "Perplexity": "https://api.perplexity.ai/chat/completions",
        "Llama": "https://api.together.xyz/v1/chat/completions"
    }
    return endpoints.get(provider, "")

def get_active_endpoint(config, key_num=1):
    """Get the active endpoint for a config - custom if set, otherwise default"""
    if key_num == 1:
        custom = config.get("custom_endpoint", "")
        return custom if custom else get_default_endpoint(config["provider"])
    else:
        custom = config.get("custom_endpoint_2", "")
        return custom if custom else get_default_endpoint(config["provider"])

def test_llm_connection(provider, api_key, endpoint):
    """Make a simple test call to validate LLM API key with specific endpoint"""
    try:
        log_to_debug(f"Testing {provider} connection at {endpoint}...")
        
        if provider == "OpenAI":
            response = requests.post(
                endpoint,
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": "gpt-3.5-turbo",
                    "messages": [{"role": "user", "content": "test"}],
                    "max_tokens": 5
                },
                timeout=5
            )
            return response.status_code in [200, 400], "Connection successful" if response.status_code == 200 else str(response.json().get("error", {}).get("message", "Error"))
        
        elif provider == "Claude":
            response = requests.post(
                endpoint,
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json"
                },
                json={
                    "model": "claude-3-haiku-20240307",
                    "max_tokens": 100,
                    "messages": [{"role": "user", "content": "test"}]
                },
                timeout=5
            )
            return response.status_code in [200, 400], "Connection successful" if response.status_code == 200 else str(response.json().get("error", {}).get("message", "Error"))
        
        elif provider == "Gemini":
            if "?key=" in endpoint:
                test_endpoint = endpoint
            else:
                test_endpoint = f"{endpoint}?key={api_key}"
            response = requests.post(
                test_endpoint,
                json={"contents": [{"parts": [{"text": "test"}]}]},
                timeout=5
            )
            return response.status_code in [200, 400], "Connection successful" if response.status_code == 200 else str(response.json().get("error", {}).get("message", "Error"))
        
        elif provider == "Mistral":
            response = requests.post(
                endpoint,
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": "mistral-small",
                    "messages": [{"role": "user", "content": "test"}],
                    "max_tokens": 5
                },
                timeout=5
            )
            return response.status_code in [200, 400], "Connection successful" if response.status_code == 200 else str(response.json().get("error", {}).get("message", "Error"))
        
        elif provider == "Deepseek":
            response = requests.post(
                endpoint,
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": "deepseek-chat",
                    "messages": [{"role": "user", "content": "test"}],
                    "max_tokens": 5
                },
                timeout=5
            )
            return response.status_code in [200, 400], "Connection successful" if response.status_code == 200 else str(response.json().get("error", {}).get("message", "Error"))
        
        elif provider == "Qwen":
            response = requests.post(
                endpoint,
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": "qwen-turbo",
                    "input": {"messages": [{"role": "user", "content": "test"}]}
                },
                timeout=5
            )
            return response.status_code in [200, 400], "Connection successful" if response.status_code == 200 else str(response.json().get("message", "Error"))
        
        elif provider == "Perplexity":
            response = requests.post(
                endpoint,
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": "llama-3.1-sonar-small-128k-online",
                    "messages": [{"role": "user", "content": "test"}]
                },
                timeout=5
            )
            return response.status_code in [200, 400], "Connection successful" if response.status_code == 200 else str(response.json().get("error", {}).get("message", "Error"))
        
        elif provider == "Llama":
            response = requests.post(
                endpoint,
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": "meta-llama/Llama-3-8b-chat-hf",
                    "messages": [{"role": "user", "content": "test"}],
                    "max_tokens": 5
                },
                timeout=5
            )
            return response.status_code in [200, 400], "Connection successful" if response.status_code == 200 else str(response.json().get("error", {}).get("message", "Error"))
        
        return False, "Unknown provider"
    
    except requests.exceptions.Timeout:
        log_to_debug(f"❌ {provider} connection timeout")
        return False, "Request timeout - check your connection"
    except Exception as e:
        log_to_debug(f"❌ {provider} error: {str(e)}")
        return False, str(e)
def authenticate_google_sheets(credentials_json):
    """
    Authenticate with Google Sheets using OAuth credentials
    Returns service object for API calls
    FIXED: Improved error handling and port management
    """
    try:
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from google.auth.transport.requests import Request
        from googleapiclient.discovery import build
        import pickle
        
        SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
        
        creds = None
        
        # Use a separate token file for Sheets
        SHEETS_TOKEN_FILE = 'token_sheets.pickle'
        
        # Check if token exists
        if os.path.exists(SHEETS_TOKEN_FILE):
            with open(SHEETS_TOKEN_FILE, 'rb') as token:
                creds = pickle.load(token)
            log_to_debug(f"✅ Loaded existing Sheets token from {SHEETS_TOKEN_FILE}")
        
        # If no valid credentials, authenticate
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    log_to_debug("🔄 Refreshing expired Sheets token...")
                    creds.refresh(Request())
                    log_to_debug("✅ Sheets token refreshed successfully")
                except Exception as refresh_error:
                    log_to_debug(f"⚠️ Token refresh failed: {str(refresh_error)}")
                    log_to_debug("🗑️ Deleting invalid token and re-authenticating...")
                    if os.path.exists(SHEETS_TOKEN_FILE):
                        os.remove(SHEETS_TOKEN_FILE)
                    creds = None
            
            if not creds or not creds.valid:
                # Save credentials to temp file
                with open('temp_credentials_sheets.json', 'w') as f:
                    json.dump(credentials_json, f)
                
                flow = InstalledAppFlow.from_client_secrets_file(
                    'temp_credentials_sheets.json', SCOPES)
                
                # ✅ FIX: Use ports that don't conflict with Gmail
                ALLOWED_PORTS = [8501, 8502, 8503, 8504, 8505, 8506]
                
                auth_success = False
                last_error = None
                
                for port in ALLOWED_PORTS:
                    try:
                        log_to_debug(f"🔑 Attempting Sheets OAuth on port {port}...")
                        creds = flow.run_local_server(
                            port=port,
                            open_browser=True,
                            authorization_prompt_message=f'Please authorize Google Sheets access in your browser (Port: {port})...',
                            success_message='✅ Authentication successful! You can close this window and return to Streamlit.',
                            redirect_uri_trailing_slash=True
                        )
                        auth_success = True
                        log_to_debug(f"✅ Sheets OAuth successful on port {port}")
                        break
                    
                    except OSError as e:
                        if "Address already in use" in str(e) or "port" in str(e).lower():
                            log_to_debug(f"⚠️ Port {port} is busy, trying next...")
                            last_error = e
                            continue
                        else:
                            raise e
                    except Exception as e:
                        log_to_debug(f"⚠️ OAuth failed on port {port}: {str(e)}")
                        last_error = e
                        continue
                
                if not auth_success:
                    error_msg = f"❌ All OAuth ports are busy or failed.\n\n"
                    error_msg += "Please ensure these redirect URIs are registered in Google Cloud Console:\n"
                    for port in ALLOWED_PORTS:
                        error_msg += f"   - http://localhost:{port}/\n"
                    error_msg += f"\nLast error: {str(last_error)}"
                    log_to_debug(error_msg)
                    raise Exception(error_msg)
                
                # Remove temp file
                if os.path.exists('temp_credentials_sheets.json'):
                    os.remove('temp_credentials_sheets.json')
            
            # Save credentials for next run
            with open(SHEETS_TOKEN_FILE, 'wb') as token:
                pickle.dump(creds, token)
            log_to_debug(f"💾 Saved Sheets token to {SHEETS_TOKEN_FILE}")
        
        service = build('sheets', 'v4', credentials=creds)
        log_to_debug("✅ Google Sheets authentication successful")
        return service
    
    except Exception as e:
        log_to_debug(f"❌ Google Sheets authentication error: {str(e)}")
        import traceback
        log_to_debug(f"📋 Full traceback:\n{traceback.format_exc()}")
        raise Exception(f"Authentication failed: {str(e)}")
    
def upload_to_google_sheets(service, df, sheet_name=None):
    """
    Upload DataFrame to Google Sheets
    Creates a new spreadsheet and returns the URL
    """
    try:
        if service is None:
            raise Exception("Service object is None - authentication may have failed")
        
        if sheet_name is None:
            sheet_name = f"Lead Report {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        log_to_debug(f"📤 Creating new Google Sheet: {sheet_name}")
        
        # Create new spreadsheet
        spreadsheet = {
            'properties': {
                'title': sheet_name
            }
        }
        
        spreadsheet = service.spreadsheets().create(
            body=spreadsheet,
            fields='spreadsheetId,spreadsheetUrl'
        ).execute()
        
        spreadsheet_id = spreadsheet.get('spreadsheetId')
        spreadsheet_url = spreadsheet.get('spreadsheetUrl')
        
        log_to_debug(f"✅ Spreadsheet created: {spreadsheet_id}")
        
        # ===== FIX: Handle NaN values and prepare data =====
        # Replace NaN with empty string and convert to proper types
        df_clean = df.copy()
        
        # Replace NaN, None, and inf values with empty string
        df_clean = df_clean.fillna('')
        
        # Replace inf and -inf with empty string
        df_clean = df_clean.replace([float('inf'), float('-inf')], '')
        
        # Convert all values to strings to avoid JSON serialization issues
        for col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str)
        
        # Replace 'nan' strings that might have been created
        df_clean = df_clean.replace('nan', '')
        df_clean = df_clean.replace('None', '')
        
        log_to_debug(f"✅ Cleaned DataFrame: removed NaN and converted to strings")
        
        # Prepare data for upload (headers + data rows)
        values = [df_clean.columns.tolist()] + df_clean.values.tolist()
        
        log_to_debug(f"📊 Prepared {len(values)} rows (including header) for upload")
        
        body = {
            'values': values
        }
        
        # Upload data
        result = service.spreadsheets().values().update(
            spreadsheetId=spreadsheet_id,
            range='Sheet1!A1',
            valueInputOption='RAW',
            body=body
        ).execute()
        
        log_to_debug(f"✅ Uploaded {result.get('updatedCells')} cells to Google Sheets")
        
        # Format header row
        requests = [
            {
                'repeatCell': {
                    'range': {
                        'sheetId': 0,
                        'startRowIndex': 0,
                        'endRowIndex': 1
                    },
                    'cell': {
                        'userEnteredFormat': {
                            'backgroundColor': {
                                'red': 0.2,
                                'green': 0.2,
                                'blue': 0.8
                            },
                            'textFormat': {
                                'foregroundColor': {
                                    'red': 1.0,
                                    'green': 1.0,
                                    'blue': 1.0
                                },
                                'bold': True
                            }
                        }
                    },
                    'fields': 'userEnteredFormat(backgroundColor,textFormat)'
                }
            },
            {
                'autoResizeDimensions': {
                    'dimensions': {
                        'sheetId': 0,
                        'dimension': 'COLUMNS',
                        'startIndex': 0,
                        'endIndex': len(df_clean.columns)
                    }
                }
            }
        ]
        
        service.spreadsheets().batchUpdate(
            spreadsheetId=spreadsheet_id,
            body={'requests': requests}
        ).execute()
        
        log_to_debug(f"✅ Applied formatting to Google Sheet")
        
        return spreadsheet_url
    
    except Exception as e:
        log_to_debug(f"❌ Error uploading to Google Sheets: {str(e)}")
        import traceback
        log_to_debug(f"📋 Full traceback:\n{traceback.format_exc()}")
        raise Exception(f"Upload failed: {str(e)}")

def get_sector_options(timeframe, region):
    """Get sector options based on timeframe and region selection"""
    if "Weekly Report" in timeframe:
        return ["ALL / GENERAL"]
    elif "Monthly Report" in timeframe and region == "India":
        return [
            "ALL / GENERAL",
            "AI",
            "B2B", 
            "B2C",
            "CLEANTECH",
            "D2C BRANDS",
            "DEEPTECH",
            "EDTECH",
            "FINTECH",
            "HEALTHCARE",
            "IOT",
            "PROPTECH",
            "SAAS"
        ]
    else:
        return [
            "All / General", 
            "Healthcare / Healthtech",
            "Real Estate / Proptech",
            "Education / Edtech",
            "AI : Artificial Intelligence"
        ]

def clear_history():
    """Clear all session state"""
    st.session_state.extracted_info = None
    st.session_state.output_logs = None
    st.session_state.df = None
    st.session_state.csv_filename = None
    st.session_state.enriched_df = None

def extract_with_apollo(df, apollo_api_key):
    """
    Extract contact information (email, phone, and designation) from LinkedIn URLs using Apollo.io API
    ✅ ENHANCED: Better phone number extraction with multiple fallbacks
    """
    import time
    import re
    
    log_to_debug(f"🔍 Starting contact extraction with Apollo API...")
    log_to_debug(f"📊 Processing {len(df)} records")
    
    # Create a copy to avoid modifying original
    enriched_df = df.copy()
    
    # Find the LinkedIn column (case-insensitive search)
    linkedin_col = None
    for col in enriched_df.columns:
        if 'linkedin' in col.lower():
            linkedin_col = col
            break
    
    if not linkedin_col:
        log_to_debug("❌ No LinkedIn column found in CSV")
        raise ValueError("CSV must contain a LinkedIn URL column (e.g., 'LinkedIn_URL', 'LinkedIn Profile URL')")
    
    log_to_debug(f"✅ Found LinkedIn column: {linkedin_col}")
    
    # Add Email, Phone, and Designation columns if they don't exist
    if 'Email' not in enriched_df.columns:
        enriched_df['Email'] = ''
    if 'Phone' not in enriched_df.columns:
        enriched_df['Phone'] = ''
    if 'Designation' not in enriched_df.columns:
        enriched_df['Designation'] = ''
    
    # Stats tracking
    total_processed = 0
    emails_found = 0
    phones_found = 0
    designations_found = 0
    errors = 0
    
    # Process each row
    for idx, row in enriched_df.iterrows():
        linkedin_url = row[linkedin_col]
        
        # Skip if no LinkedIn URL
        if pd.isna(linkedin_url) or not linkedin_url or linkedin_url == "Profile not found":
            log_to_debug(f"  ⏭️ Row {idx+1}: No LinkedIn URL - skipping")
            continue
        
        total_processed += 1
        log_to_debug(f"  🔍 Row {idx+1}/{len(enriched_df)}: Processing {linkedin_url}")
        
        # Call Apollo API
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
                
                # Extract email
                email = person.get('email', '')
                email_status = person.get('email_status', '')
                
                # ✅ ENHANCED: Extract phone with multiple fallbacks (SAME AS LINKEDIN SCRAPER)
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
                
                # Clean phone number
                if phone:
                    phone = phone.strip()
                
                # ✅ Extract designation/title
                title = person.get('title', '')
                
                # Update dataframe
                if email:
                    enriched_df.at[idx, 'Email'] = email
                    emails_found += 1
                    status_icon = "✓" if email_status == 'verified' else "~"
                    log_to_debug(f"    {status_icon} Email: {email[:30]}... ({email_status})")
                
                if phone:
                    enriched_df.at[idx, 'Phone'] = phone
                    phones_found += 1
                    log_to_debug(f"    ✓ Phone: {phone}")
                
                if title:
                    enriched_df.at[idx, 'Designation'] = title
                    designations_found += 1
                    log_to_debug(f"    ✓ Designation: {title}")
                
                if not email and not phone and not title:
                    log_to_debug(f"    ✗ No contact info found")
            
            elif response.status_code == 404:
                log_to_debug(f"    ✗ Profile not found in Apollo database")
            
            elif response.status_code == 429:
                log_to_debug(f"    ⚠️ Rate limit hit, waiting 5s...")
                time.sleep(5)
                continue
            
            elif response.status_code == 403:
                log_to_debug(f"    ❌ Apollo 403 Forbidden - Check API key")
                errors += 1
            
            elif response.status_code == 401:
                log_to_debug(f"    ❌ Apollo: Invalid API key")
                raise ValueError("Invalid Apollo API key - please check your credentials")
            
            else:
                log_to_debug(f"    ✗ Apollo API error: {response.status_code}")
                errors += 1
        
        except requests.exceptions.Timeout:
            log_to_debug(f"    ✗ Request timeout")
            errors += 1
        
        except requests.exceptions.RequestException as e:
            log_to_debug(f"    ✗ Request failed: {str(e)[:100]}")
            errors += 1
        
        except Exception as e:
            log_to_debug(f"    ✗ Unexpected error: {str(e)[:100]}")
            errors += 1
        
        # Rate limiting: sleep between requests
        time.sleep(0.5)
    
    # Final summary
    log_to_debug(f"\n📊 CONTACT EXTRACTION SUMMARY:")
    log_to_debug(f"  • Total records: {len(enriched_df)}")
    log_to_debug(f"  • Processed: {total_processed}")
    log_to_debug(f"  • Emails found: {emails_found}")
    log_to_debug(f"  • Phones found: {phones_found}")
    log_to_debug(f"  • Designations found: {designations_found}")
    log_to_debug(f"  • Errors: {errors}")
    if total_processed > 0:
        log_to_debug(f"  • Email success rate: {(emails_found/total_processed*100):.1f}%")
        log_to_debug(f"  • Phone success rate: {(phones_found/total_processed*100):.1f}%")
        log_to_debug(f"  • Designation success rate: {(designations_found/total_processed*100):.1f}%")
    
    log_to_debug(f"✅ Contact extraction completed")
    return enriched_df

def initialize_leads_database():
    """Initialize SQLite database for lead tracking"""
    try:
        conn = sqlite3.connect("leads.db")
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS leads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL,
                founder_name TEXT,
                company_name TEXT,
                sent_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                opened_at TIMESTAMP,
                replied INTEGER DEFAULT 0,
                bounced INTEGER DEFAULT 0,
                template_used TEXT,
                sender_email TEXT,
                thread_id TEXT,
                last_checked TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
        log_to_debug("✅ Leads database initialized")
        
        # Update schema if needed
        update_database_schema()
    except Exception as e:
        log_to_debug(f"❌ Error initializing database: {str(e)}")

def update_database_schema():
    """Update database schema to add thread_id column if it doesn't exist"""
    try:
        conn = sqlite3.connect("leads.db")
        cursor = conn.cursor()
        
        # Check if thread_id column exists
        cursor.execute("PRAGMA table_info(leads)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'thread_id' not in columns:
            cursor.execute("ALTER TABLE leads ADD COLUMN thread_id TEXT")
            log_to_debug("✅ Added thread_id column to database")
        
        if 'last_checked' not in columns:
            cursor.execute("ALTER TABLE leads ADD COLUMN last_checked TIMESTAMP")
            log_to_debug("✅ Added last_checked column to database")
        
        conn.commit()
        conn.close()
    except Exception as e:
        log_to_debug(f"❌ Error updating database schema: {str(e)}")

def load_leads_data():
    """Load leads data from SQLite database"""
    try:
        conn = sqlite3.connect("leads.db")
        df = pd.read_sql_query("SELECT * FROM leads ORDER BY sent_at DESC", conn)
        conn.close()
        return df
    except Exception as e:
        log_to_debug(f"❌ Error loading leads data: {str(e)}")
        return pd.DataFrame()

def log_email_to_database(recipient_email, founder_name, company_name, template_used, sender_email, success, thread_id=None):
    """
    Log sent email to database with thread_id
    """
    try:
        conn = sqlite3.connect("leads.db")
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO leads (email, founder_name, company_name, template_used, sender_email, bounced, thread_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (recipient_email, founder_name, company_name, template_used, sender_email, 0 if success else 1, thread_id))
        
        conn.commit()
        conn.close()
        
        if thread_id:
            log_to_debug(f"✅ Logged to database: {recipient_email} | Thread: {thread_id}")
        else:
            log_to_debug(f"⚠️ Logged to database: {recipient_email} | No thread_id")
    except Exception as e:
        log_to_debug(f"❌ Error logging to database: {str(e)}")
# ============ MAIL SENDER FUNCTIONS ============

def send_email_via_gmail(sender_email, oauth_credentials, recipient_email, subject, body, lead_id=None):
    """
    Send email using Gmail API with OAuth credentials and tracking pixel
    ✅ FIXED: Complete null/NoneType validation
    Returns: (success: bool, message: str, thread_id: str)
    """
    try:
        # ✅ CRITICAL FIX 1: Validate ALL inputs at the start
        # Validate recipient email
        if recipient_email is None or pd.isna(recipient_email):
            return False, "Invalid recipient email: None or NaN", None
        
        recipient_email = str(recipient_email).strip()
        
        if not recipient_email or '@' not in recipient_email:
            return False, f"Invalid recipient email format: {recipient_email}", None
        
        # ✅ CRITICAL FIX 2: Validate subject
        if subject is None or pd.isna(subject):
            subject = "Hello"
        else:
            subject = str(subject).strip()
            if not subject:
                subject = "Hello"
        
        # ✅ CRITICAL FIX 3: Validate body
        if body is None or pd.isna(body):
            body = "Hello, how are you?"
        else:
            body = str(body).strip()
            if not body:
                body = "Hello, how are you?"
        
        # ✅ CRITICAL FIX 4: Validate sender_email
        if sender_email is None or pd.isna(sender_email):
            return False, "Invalid sender email: None or NaN", None
        
        sender_email = str(sender_email).strip()
        
        if not sender_email or '@' not in sender_email:
            return False, f"Invalid sender email format: {sender_email}", None
        
        # ✅ CRITICAL FIX 5: Validate oauth_credentials
        if oauth_credentials is None:
            return False, "OAuth credentials not provided", None
        
        SCOPES = ['https://www.googleapis.com/auth/gmail.send', 
                  'https://www.googleapis.com/auth/gmail.readonly']
        
        creds = None
        token_file = f'token_gmail_{sender_email.replace("@", "_").replace(".", "_")}.pickle'
        
        # Check if token exists for this email
        if os.path.exists(token_file):
            with open(token_file, 'rb') as token:
                creds = pickle.load(token)
        
        # If no valid credentials, authenticate
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                # Save credentials to temp file
                with open('temp_credentials_gmail.json', 'w') as f:
                    json.dump(oauth_credentials, f)
                
                flow = InstalledAppFlow.from_client_secrets_file(
                    'temp_credentials_gmail.json', SCOPES)
                
                # Try multiple fixed ports in sequence
                ports_to_try = [8080, 8081, 8082, 8083, 8084, 8085]
                auth_success = False
                
                for port in ports_to_try:
                    try:
                        log_to_debug(f"🔑 Attempting OAuth on port {port}...")
                        creds = flow.run_local_server(
                            port=port,
                            open_browser=True,
                            authorization_prompt_message=f'Please authorize Gmail access in your browser (Port: {port})...',
                            success_message='✅ Authentication successful! You can close this window and return to Streamlit.'
                        )
                        auth_success = True
                        log_to_debug(f"✅ OAuth successful on port {port}")
                        break
                    
                    except OSError as e:
                        log_to_debug(f"⚠️ Port {port} unavailable, trying next port...")
                        continue
                
                if not auth_success:
                    raise Exception("❌ All OAuth ports (8080-8085) are busy. Please close some applications and try again.")
                
                # Remove temp file
                if os.path.exists('temp_credentials_gmail.json'):
                    os.remove('temp_credentials_gmail.json')
            
            # Save credentials for next use
            with open(token_file, 'wb') as token:
                pickle.dump(creds, token)
        
        # Build Gmail service
        service = build('gmail', 'v1', credentials=creds)
        
        # Create message with HTML support for tracking pixel
        message = MIMEMultipart('alternative')
        message['to'] = recipient_email
        message['from'] = sender_email
        message['subject'] = subject
        
        # Convert plain text body to HTML with proper line breaks
        html_body_text = body.replace('\n', '<br>')
        
        # ✅ CRITICAL FIX: Improved tracking pixel implementation
        if lead_id:
            current_tracking_url = st.session_state.get('tracking_server_url', TRACKING_SERVER_URL)
            
            # ✅ Validate tracking URL
            if current_tracking_url is None or pd.isna(current_tracking_url):
                current_tracking_url = TRACKING_SERVER_URL
            else:
                current_tracking_url = str(current_tracking_url).strip()
            
            tracking_url = f"{current_tracking_url}/track/{lead_id}"
    
            html_body = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto; padding: 20px;">
    <div style="white-space: pre-wrap;">{html_body_text}</div>
    
    <!-- Tracking Pixel -->
    <img src="{tracking_url}" width="1" height="1" style="display:none;" alt="" />
</body>
</html>
"""
            
            # Add both plain text and HTML parts
            part_text = MIMEText(body, 'plain', 'utf-8')
            part_html = MIMEText(html_body, 'html', 'utf-8')
            
            message.attach(part_text)
            message.attach(part_html)
            
            log_to_debug(f"📧 Added tracking pixel for lead_id: {lead_id}")
            log_to_debug(f"   Tracking URL: {tracking_url}")
        else:
            # No tracking - just plain text
            part = MIMEText(body, 'plain', 'utf-8')
            message.attach(part)
        
        # Encode message
        raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode('utf-8')
        
        # Send message
        send_message = service.users().messages().send(
            userId='me',
            body={'raw': raw_message}
        ).execute()
        
        # Get thread ID from sent message
        thread_id = send_message.get('threadId', '')
        message_id = send_message.get('id', '')
        
        if thread_id:
            log_to_debug(f"✅ Email sent to {recipient_email} | Message ID: {message_id} | Thread ID: {thread_id}")
            return True, f"Email sent successfully (ID: {message_id})", thread_id
        else:
            log_to_debug(f"⚠️ Email sent to {recipient_email} but no thread_id returned | Message ID: {message_id}")
            return True, f"Email sent successfully (ID: {message_id})", None
    
    except Exception as e:
        error_msg = f"Failed to send email: {str(e)}"
        log_to_debug(f"❌ {error_msg}")
        
        # ✅ Add detailed traceback for debugging
        import traceback
        log_to_debug(f"📋 Full traceback:\n{traceback.format_exc()}")
        
        return False, error_msg, None
    
# Replace your send_emails_batch function with this fixed version:

def send_emails_batch(mail_config, recipients_df):
    """
    Send emails to all recipients in batch with tracking pixels
    ✅ FIXED: Complete email validation and error handling
    """
    import time
    
    results = {
        'sent': 0,
        'failed': 0,
        'errors': []
    }
    
    valid_senders = [(i, email) for i, email in enumerate(mail_config["dummy_mail_addresses"]) 
                     if email.strip() and mail_config["oauth_credentials"][i] is not None]
    
    if not valid_senders:
        log_to_debug("❌ No valid senders configured")
        return results
    
    total_recipients = len(recipients_df)
    log_to_debug(f"📧 Starting batch email send: {total_recipients} recipients, {len(valid_senders)} senders")
    
    for idx, row in recipients_df.iterrows():
        try:


            # ✅ CRITICAL FIX 1: Validate recipient email BEFORE processing
            recipient_email_raw = row.get('Email', '')
            
            # Convert to string and handle None/NaN
            if recipient_email_raw is None or pd.isna(recipient_email_raw):
                log_to_debug(f"⏭️ Row {idx+1}: Email is None/NaN - skipping")
                continue
            
            recipient_email = str(recipient_email_raw).strip()
            
            if not recipient_email or '@' not in recipient_email:
                log_to_debug(f"⏭️ Row {idx+1}: Invalid email format ({recipient_email}) - skipping")
                continue
            
            if '@' not in recipient_email:
                log_to_debug(f"⏭️ Row {idx+1}: Malformed email ({recipient_email}) - skipping")
                continue
            
            # Select sender (round-robin)
            sender_idx, sender_email = valid_senders[idx % len(valid_senders)]
            oauth_creds = mail_config["oauth_credentials"][sender_idx]


            # ✅ FIX 2: Safe null handling for founder_name and company_name
            founder_name_raw = row.get('Founder_Name', None)
            if founder_name_raw is None or pd.isna(founder_name_raw) or str(founder_name_raw).strip() == '':
                founder_name = 'Valued Recipient'
            else:
                founder_name = str(founder_name_raw).strip()
            
            company_name_raw = row.get('Company_Name', None)
            if company_name_raw is None or pd.isna(company_name_raw) or str(company_name_raw).strip() == '':
                company_name = 'Unknown'
            else:
                company_name = str(company_name_raw).strip()
            
            # Determine template based on industry
            if 'Industry_Sector' in recipients_df.columns:
                industry = row.get('Industry_Sector', '')
                if not pd.isna(industry):
                    industry_lower = str(industry).lower()
                    if 'health' in industry_lower or 'medical' in industry_lower:
                        template_key = 'healthtech'
                    elif 'edtech' in industry_lower or 'education' in industry_lower:
                        template_key = 'edtech'
                    elif 'real estate' in industry_lower or 'proptech' in industry_lower:
                        template_key = 'real_estate'
                    elif 'ai' in industry_lower or 'tech' in industry_lower:
                        template_key = 'ai'
                    else:
                        template_key = mail_config["selected_template"]
                else:
                    template_key = mail_config["selected_template"]
            else:
                template_key = mail_config["selected_template"]
            

            # Get template
            if template_key == "customize":
                template_text = mail_config.get("custom_template", "")
            else:
                template_text = mail_config["permanent_templates"].get(
                    template_key, 
                    EMAIL_TEMPLATES.get(template_key, "")
                )
            
            # ✅ FIX 3: Validate template before personalization
            if not template_text or template_text is None or pd.isna(template_text) or str(template_text).strip() == '':
                # Fallback to default template
                template_text = EMAIL_TEMPLATES.get(template_key, "Subject: Hello\n\nDear {name},\n\nBest regards,\n{sender_name}")
            
            # Personalize email
            personalized_email = safe_personalize_email(
                template_text,
                founder_name,
                company_name,
                "Your Team"
            )
            

            # ✅ FIX 4: Safe subject/body extraction
            if personalized_email is None or pd.isna(personalized_email):
                subject = "Hello"
                body = "Hello, how are you?"
            else:
                personalized_email_str = str(personalized_email).strip()
                
                if not personalized_email_str:
                    subject = "Hello"
                    body = "Hello, how are you?"
                else:
                    lines = personalized_email_str.split('\n')
                    
                    subject = ""
                    body = personalized_email_str
                    
                    for line in lines:
                        if line and 'Subject:' in line:
                            subject = line.replace('Subject:', '').strip()
                            body = '\n'.join(lines[lines.index(line)+1:]).strip()
                            break
                    
                    if not subject or subject.strip() == '':
                        subject = "Hello"
                    if not body or body.strip() == '':
                        body = "Hello, how are you?"
            
            # ✅ CRITICAL FIX: Create lead record FIRST to get lead_id
            conn = sqlite3.connect("leads.db")
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO leads (email, founder_name, company_name, template_used, sender_email, bounced)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (recipient_email, founder_name, company_name, template_key, sender_email, 0))
            
            lead_id = cursor.lastrowid  # Get the auto-generated ID
            conn.commit()
            conn.close()
            
            log_to_debug(f"📝 Created lead record with ID: {lead_id}")
            log_to_debug(f"   Email: {recipient_email}")
            log_to_debug(f"   Founder: {founder_name}")
            
            # ✅ CRITICAL: Send email WITH the lead_id for tracking
            success, message, thread_id = send_email_via_gmail(
                sender_email=sender_email,
                oauth_credentials=oauth_creds,
                recipient_email=recipient_email,
                subject=subject,
                body=body,
                lead_id=lead_id  # ✅ Pass the lead_id here!
            )
            
            if success:
                results['sent'] += 1
                
                # Update the record with thread_id
                conn = sqlite3.connect("leads.db")
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE leads 
                    SET thread_id = ?
                    WHERE id = ?
                """, (thread_id if thread_id else None, lead_id))
                conn.commit()
                conn.close()
                
                log_to_debug(f"✅ [{results['sent']}/] Email sent successfully!")
                log_to_debug(f"   To: {recipient_email}")
                log_to_debug(f"   Lead ID: {lead_id}")
                log_to_debug(f"   Thread ID: {thread_id}")
            else:
                results['failed'] += 1
                results['errors'].append({
                    'recipient': recipient_email,
                    'error': message
                })
                
                # Mark as bounced
                conn = sqlite3.connect("leads.db")
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE leads 
                    SET bounced = 1
                    WHERE id = ?
                """, (lead_id,))
                conn.commit()
                conn.close()
                
                log_to_debug(f"❌ Failed to send to {recipient_email}: {message}")
            
            # Rate limiting: wait between emails
            time.sleep(1)
        
        except Exception as e:
            results['failed'] += 1
            results['errors'].append({
                'recipient': row.get('Email', 'Unknown'),
                'error': str(e)
            })
            log_to_debug(f"❌ Error sending to {row.get('Email', 'Unknown')}: {str(e)}")
    
    log_to_debug(f"📊 Batch send complete: {results['sent']} sent, {results['failed']} failed")
    return results
def safe_personalize_email(template, name, company, sender_name="Your Team"):
    """
    Safely personalize email template, handling NaN, None, and all data types
    ✅ FIXED: Complete null handling with template validation
    """
    try:
        # ✅ FIX 1: Validate template exists and is not None
        if template is None or pd.isna(template):
            template = "Subject: Hello\n\nDear {name},\n\nBest regards,\n{sender_name}"
        
        template = str(template).strip()
        
        if not template:
            template = "Subject: Hello\n\nDear {name},\n\nBest regards,\n{sender_name}"
        
        # ✅ FIX 2: Convert all values to safe strings with comprehensive null checks
        # Handle name
        if name is None or pd.isna(name) or str(name).strip() == '' or str(name).lower() == 'none':
            name_str = "Valued Recipient"
        else:
            name_str = str(name).strip()
        
        # Handle company
        if company is None or pd.isna(company) or str(company).strip() == '' or str(company).lower() == 'none':
            company_str = "Your Company"
        else:
            company_str = str(company).strip()
        
        # Handle sender_name
        if sender_name is None or pd.isna(sender_name) or str(sender_name).strip() == '' or str(sender_name).lower() == 'none':
            sender_str = "Your Team"
        else:
            sender_str = str(sender_name).strip()
        
        # ✅ FIX 3: Replace ALL placeholder variations
        result = template.replace("{name}", name_str)
        result = result.replace("{Founder_Name}", name_str)
        result = result.replace("{NAME}", name_str)
        result = result.replace("{founder_name}", name_str)
        
        result = result.replace("{company_name}", company_str)
        result = result.replace("{Company_Name}", company_str)
        result = result.replace("{COMPANY_NAME}", company_str)
        
        result = result.replace("{sender_name}", sender_str)
        result = result.replace("{Sender_Name}", sender_str)
        result = result.replace("{SENDER_NAME}", sender_str)
        
        return result
    
    except Exception as e:
        log_to_debug(f"❌ Error in safe_personalize_email: {str(e)}")
        # Return a safe fallback template
        return f"Subject: Hello\n\nDear Valued Recipient,\n\nBest regards,\nYour Team"


def check_email_replies():
    """
    Check for replies to sent emails using Gmail API
    Updates the database with reply status
    FIXED VERSION - Preserves opened_at timestamp + Auto-detects scope issues
    """
    try:
        conn = sqlite3.connect("leads.db")
        cursor = conn.cursor()
        
        # Get all emails that haven't been replied to yet and have a thread_id
        cursor.execute("""
            SELECT id, email, sender_email, thread_id, opened_at
            FROM leads 
            WHERE replied = 0 
            AND thread_id IS NOT NULL 
            AND thread_id != ''
            ORDER BY sent_at DESC
        """)
        
        pending_emails = cursor.fetchall()
        
        if not pending_emails:
            log_to_debug("📭 No pending emails to check for replies")
            conn.close()
            return 0
        
        log_to_debug(f"🔍 Checking {len(pending_emails)} emails for replies...")
        
        replies_found = 0
        
        # Group by sender email to minimize authentication
        emails_by_sender = {}
        for row in pending_emails:
            lead_id, recipient_email, sender_email, thread_id, current_opened_at = row
            if sender_email not in emails_by_sender:
                emails_by_sender[sender_email] = []
            emails_by_sender[sender_email].append((lead_id, recipient_email, thread_id, current_opened_at))
        
        log_to_debug(f"📊 Found {len(emails_by_sender)} unique sender accounts")
        
        # Check each sender's emails
        for sender_email, emails in emails_by_sender.items():
            try:
                # ✅ CRITICAL FIX: Use BOTH scopes (same as send_email_via_gmail)
                SCOPES = ['https://www.googleapis.com/auth/gmail.send',
                          'https://www.googleapis.com/auth/gmail.readonly']
                
                token_file = f'token_gmail_{sender_email.replace("@", "_").replace(".", "_")}.pickle'
                
                if not os.path.exists(token_file):
                    log_to_debug(f"⚠️ No token file found for {sender_email}, skipping...")
                    continue
                
                with open(token_file, 'rb') as token:
                    creds = pickle.load(token)
                
                # ✅ NEW: Validate token has required scopes
                required_scope = 'https://www.googleapis.com/auth/gmail.readonly'
                
                if not creds.scopes or required_scope not in creds.scopes:
                    log_to_debug(f"❌ Token for {sender_email} missing required scope: {required_scope}")
                    log_to_debug(f"   Current scopes: {creds.scopes}")
                    log_to_debug(f"   🗑️ Deleting invalid token file...")
                    
                    # Delete the invalid token
                    try:
                        os.remove(token_file)
                        log_to_debug(f"   ✅ Token deleted successfully")
                    except Exception as e:
                        log_to_debug(f"   ⚠️ Could not delete token: {str(e)}")
                    
                    log_to_debug(f"   ⚠️ Action required: Send an email from {sender_email} to re-authenticate")
                    log_to_debug(f"   💡 The new token will have both 'gmail.send' and 'gmail.readonly' scopes")
                    continue
                
                # Refresh if expired
                if creds and creds.expired and creds.refresh_token:
                    try:
                        creds.refresh(Request())
                        with open(token_file, 'wb') as token:
                            pickle.dump(creds, token)
                        log_to_debug(f"✅ Refreshed expired token for {sender_email}")
                    except Exception as e:
                        log_to_debug(f"⚠️ Failed to refresh token for {sender_email}: {str(e)}")
                        log_to_debug(f"   Deleting token and requiring re-authentication...")
                        try:
                            os.remove(token_file)
                        except:
                            pass
                        continue
                
                service = build('gmail', 'v1', credentials=creds)
                log_to_debug(f"✅ Gmail service authenticated for {sender_email}")
                
                # Check each thread
                for lead_id, recipient_email, thread_id, current_opened_at in emails:
                    try:
                        log_to_debug(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
                        log_to_debug(f"🔍 Checking thread: {thread_id}")
                        log_to_debug(f"   Recipient: {recipient_email}")
                        log_to_debug(f"   Sender: {sender_email}")
                        log_to_debug(f"   Current opened_at: {current_opened_at if current_opened_at else 'Not opened yet'}")
                        
                        # Get thread messages
                        thread = service.users().threads().get(
                            userId='me',
                            id=thread_id,
                            format='full'
                        ).execute()
                        
                        messages = thread.get('messages', [])
                        log_to_debug(f"   📧 Thread has {len(messages)} message(s)")
                        
                        # Log ALL messages in thread with details
                        for msg_idx, msg in enumerate(messages, start=1):
                            headers = msg.get('payload', {}).get('headers', [])
                            
                            from_header = None
                            to_header = None
                            subject_header = None
                            date_header = None
                            
                            for header in headers:
                                if header['name'].lower() == 'from':
                                    from_header = header['value']
                                elif header['name'].lower() == 'to':
                                    to_header = header['value']
                                elif header['name'].lower() == 'subject':
                                    subject_header = header['value']
                                elif header['name'].lower() == 'date':
                                    date_header = header['value']
                            
                            log_to_debug(f"   ╔═══ Message #{msg_idx} ═══")
                            log_to_debug(f"   ║ FROM: {from_header}")
                            log_to_debug(f"   ║ TO: {to_header}")
                            log_to_debug(f"   ║ SUBJECT: {subject_header}")
                            log_to_debug(f"   ║ DATE: {date_header}")
                            log_to_debug(f"   ╚═══════════════════════")
                        
                        # Check if there are more than 1 message (original + reply)
                        if len(messages) > 1:
                            # Iterate through messages starting from the second one
                            has_reply = False
                            reply_date = None
                            
                            for msg_idx, msg in enumerate(messages[1:], start=2):
                                headers = msg.get('payload', {}).get('headers', [])
                                
                                from_header = None
                                date_header = None
                                
                                for header in headers:
                                    if header['name'].lower() == 'from':
                                        from_header = header['value']
                                    elif header['name'].lower() == 'date':
                                        date_header = header['value']
                                
                                # Extract email from "Name <email@domain.com>" format
                                if from_header:
                                    import re
                                    email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', from_header.lower())
                                    sender_in_header = email_match.group(0) if email_match else from_header.lower()
                                    
                                    log_to_debug(f"   🔎 Comparing:")
                                    log_to_debug(f"      Recipient email: {recipient_email.lower()}")
                                    log_to_debug(f"      Extracted from header: {sender_in_header}")
                                    
                                    # Check if this message is FROM the recipient
                                    if recipient_email.lower() == sender_in_header or recipient_email.lower() in sender_in_header:
                                        has_reply = True
                                        reply_date = date_header
                                        log_to_debug(f"   ✅ ✅ ✅ REPLY DETECTED! ✅ ✅ ✅")
                                        log_to_debug(f"      Message FROM: {from_header}")
                                        log_to_debug(f"      Reply date: {reply_date}")
                                        break
                            
                            if has_reply:
                                # ✅ FIX: Only update replied status, preserve opened_at if it exists
                                if current_opened_at:
                                    # Already opened via tracking pixel - preserve that timestamp
                                    cursor.execute("""
                                        UPDATE leads 
                                        SET replied = 1,
                                            last_checked = CURRENT_TIMESTAMP 
                                        WHERE id = ?
                                    """, (lead_id,))
                                    log_to_debug(f"   💾 Reply saved! Preserved existing opened_at: {current_opened_at}")
                                else:
                                    # Not opened yet - use reply date as opened date
                                    if reply_date:
                                        cursor.execute("""
                                            UPDATE leads 
                                            SET replied = 1, 
                                                opened_at = ?,
                                                last_checked = CURRENT_TIMESTAMP 
                                            WHERE id = ?
                                        """, (reply_date, lead_id))
                                        log_to_debug(f"   💾 Reply saved! Set opened_at to reply date: {reply_date}")
                                    else:
                                        cursor.execute("""
                                            UPDATE leads 
                                            SET replied = 1, 
                                                opened_at = CURRENT_TIMESTAMP,
                                                last_checked = CURRENT_TIMESTAMP 
                                            WHERE id = ?
                                        """, (lead_id,))
                                        log_to_debug(f"   💾 Reply saved! Set opened_at to current time")
                                
                                conn.commit()
                                replies_found += 1
                                
                            else:
                                cursor.execute("""
                                    UPDATE leads 
                                    SET last_checked = CURRENT_TIMESTAMP 
                                    WHERE id = ?
                                """, (lead_id,))
                                conn.commit()
                                log_to_debug(f"   ⏭️ No reply detected (checked {len(messages)-1} messages)")
                                
                        else:
                            cursor.execute("""
                                UPDATE leads 
                                SET last_checked = CURRENT_TIMESTAMP 
                                WHERE id = ?
                            """, (lead_id,))
                            conn.commit()
                            log_to_debug(f"   📭 Only 1 message - no replies yet")
                        
                        # Small delay
                        import time
                        time.sleep(0.2)
                    
                    except Exception as e:
                        log_to_debug(f"⚠️ Error checking thread {thread_id}: {str(e)}")
                        try:
                            cursor.execute("""
                                UPDATE leads 
                                SET last_checked = CURRENT_TIMESTAMP 
                                WHERE id = ?
                            """, (lead_id,))
                            conn.commit()
                        except:
                            pass
                        continue
            
            except Exception as e:
                log_to_debug(f"❌ Error for {sender_email}: {str(e)}")
                import traceback
                log_to_debug(f"📋 Traceback: {traceback.format_exc()}")
                continue
        
        conn.close()
        
        log_to_debug(f"✅ Reply check complete: {replies_found} new replies found")
        return replies_found
    
    except Exception as e:
        log_to_debug(f"❌ Error checking replies: {str(e)}")
        import traceback
        log_to_debug(f"📋 Full traceback: {traceback.format_exc()}")
        return 0
    
def render_mail_sender_section(tab_name=""):
    """
    Render the mail sender section that appears in all tabs
    Uses st.session_state.enriched_df from Apollo extraction or local CSV
    """
    st.markdown("---")
    st.markdown(f"### 📤 Send Mails ")
    st.caption("Configure and send personalized emails to your leads")

    # Initialize mail sender config in session state
    if "mail_sender_config" not in st.session_state:
        st.session_state.mail_sender_config = {
            "num_dummy_mails": 1,
            "dummy_mail_addresses": [""],
            "oauth_credentials": [None],
            "selected_template": "healthtech",
            "custom_template": EMAIL_TEMPLATES["customize"],
            "template_edit_mode": False,
            "temp_template": "",
            "permanent_templates": EMAIL_TEMPLATES.copy(),
            "auto_template_mapping": True
        }

    mail_config = st.session_state.mail_sender_config

    # Recipient data source selection
    st.markdown("#### 📊 Recipient Data Source")

    data_source = st.radio(
        "Choose recipient data source",
        options=["📂 Use Generated Data from Apollo", "📁 Upload Local CSV"],
        key=f"data_source_{tab_name}",
        horizontal=True,
        help="Choose to use data from Apollo extraction or upload your own CSV file"
    )

    st.markdown("---")

    # Initialize variables
    has_recipients = False
    recipients_df = None

    if data_source == "📁 Upload Local CSV":
        # Local CSV upload option
        st.markdown("##### 📤 Upload Your Recipient CSV")

        local_csv = st.file_uploader(
            "Upload CSV with recipient data",
            type=["csv"],
            key=f"local_csv_{tab_name}",
            help="CSV must contain 'Email' and 'Founder_Name' columns. Optional: 'Industry_Sector' column for auto-template mapping"
        )

        if local_csv:
            try:
                local_df = pd.read_csv(local_csv)

                # Validate required columns
                missing_cols = []
                if 'Email' not in local_df.columns:
                    missing_cols.append('Email')
                if 'Founder_Name' not in local_df.columns:
                    missing_cols.append('Founder_Name')

                if missing_cols:
                    st.error(f"❌ CSV must contain: {', '.join(missing_cols)}")
                    has_recipients = False
                else:
                    recipients_df = local_df
                    has_recipients = True
                    st.success(f"✅ Loaded {len(local_df)} recipients from local CSV")

                    with st.expander("👀 Preview Uploaded Data"):
                        st.dataframe(local_df.head(10), width='stretch')

                        # Show column info
                        st.write("**Columns detected:**")
                        cols_info = ", ".join(local_df.columns.tolist())
                        st.info(f"📋 {cols_info}")

                        # Check for Industry column
                        if 'Industry_Sector' in local_df.columns:
                            st.write("**Industry Distribution:**")
                            st.write(local_df['Industry_Sector'].value_counts())

            except Exception as e:
                st.error(f"❌ Error reading CSV: {str(e)}")
                has_recipients = False
        else:
            st.info("💡 Please upload a CSV file to proceed")

    else:
        # Use enriched_df from Apollo (existing logic)
        has_recipients = hasattr(st.session_state, 'enriched_df') and st.session_state.enriched_df is not None

        if has_recipients:
            recipients_df = st.session_state.enriched_df
        else:
            st.warning("⚠️ No Apollo data found. Please use Apollo Extract in Tab 1 first, or switch to 'Upload Local CSV' option.")

    st.markdown("---")

    # Sender Configuration
    st.markdown("#### 📮 Sender Email Configuration")

    num_mails = st.number_input(
        "Number of Sender Email Accounts",
        min_value=1,
        max_value=10,
        value=mail_config["num_dummy_mails"],
        key=f"num_dummy_mails_{tab_name}",
        help="Enter how many email accounts you want to rotate for sending"
    )

    # Update if number changed
    if num_mails != mail_config["num_dummy_mails"]:
        mail_config["num_dummy_mails"] = num_mails
        if num_mails > len(mail_config["dummy_mail_addresses"]):
            mail_config["dummy_mail_addresses"].extend([""] * (num_mails - len(mail_config["dummy_mail_addresses"])))
            mail_config["oauth_credentials"].extend([None] * (num_mails - len(mail_config["oauth_credentials"])))
        else:
            mail_config["dummy_mail_addresses"] = mail_config["dummy_mail_addresses"][:num_mails]
            mail_config["oauth_credentials"] = mail_config["oauth_credentials"][:num_mails]

    st.markdown("---")

    # OAuth Credentials
    st.markdown("#### 🔐 Gmail OAuth Credentials")
    st.caption(f"Upload {num_mails} OAuth JSON file(s) - one for each sender email")

    for i in range(num_mails):
        with st.expander(f"📁 OAuth Credentials for Sender #{i+1}", expanded=False):
            oauth_file = st.file_uploader(
                f"Upload OAuth JSON for Sender #{i+1}",
                type=["json"],
                key=f"oauth_uploader_{tab_name}_{i}",
                help="Upload Google OAuth 2.0 Client ID JSON file"
            )

            if oauth_file:
                try:
                    oauth_data = json.load(oauth_file)
                    if "installed" in oauth_data or "web" in oauth_data:
                        mail_config["oauth_credentials"][i] = oauth_data
                        st.success(f"✅ OAuth loaded for Sender #{i+1}")
                        log_to_debug(f"OAuth credentials loaded for sender slot {i+1}")
                    else:
                        st.error("❌ Invalid OAuth JSON structure")
                except:
                    st.error("❌ Error reading OAuth file")

    st.markdown("---")

    # Sender Email Addresses
    st.markdown("#### 📧 Sender Email Addresses")
    st.caption("Enter the Gmail addresses corresponding to the OAuth credentials")

    cols = st.columns(2)
    for i in range(num_mails):
        col_idx = i % 2
        with cols[col_idx]:
            email = st.text_input(
                f"Sender Email #{i+1}",
                value=mail_config["dummy_mail_addresses"][i],
                key=f"dummy_email_{tab_name}_{i}",
                placeholder="sender@gmail.com"
            )
            mail_config["dummy_mail_addresses"][i] = email

    st.markdown("---")

    # Template Configuration
    st.markdown("#### 📝 Email Template Configuration")

    # Industry to template mapping function
    def map_industry_to_template(industry_str):
        """Map industry sector to appropriate template"""
        if not industry_str or pd.isna(industry_str):
            return mail_config["selected_template"]

        industry_lower = str(industry_str).lower().strip()

        # Define mapping
        industry_map = {
            "healthcare": "healthtech",
            "health": "healthtech",
            "medical": "healthtech",
            "pharma": "healthtech",
            "biotech": "healthtech",
            "edtech": "edtech",
            "education": "edtech",
            "learning": "edtech",
            "real estate": "real_estate",
            "property": "real_estate",
            "proptech": "real_estate",
            "construction": "real_estate",
            "ai": "ai",
            "artificial intelligence": "ai",
            "machine learning": "ai",
            "technology": "ai",
            "software": "ai",
            "tech": "ai"
        }

        # Check each keyword
        for keyword, template in industry_map.items():
            if keyword in industry_lower:
                return template

        return mail_config["selected_template"]

    # Show auto-mapping info
    if has_recipients and recipients_df is not None and 'Industry_Sector' in recipients_df.columns:
        st.info("🤖 Auto-template mapping ENABLED - Templates will be selected based on Industry_Sector column")

        with st.expander("🔍 View Industry → Template Mapping"):
            industry_template_map = {
                "Healthcare": "healthtech",
                "Edtech": "edtech",
                "Real Estate/PropTech": "real_estate",
                "AI/Technology": "ai"
            }

            st.write("**Mapping Rules:**")
            for industry, template in industry_template_map.items():
                st.write(f"• {industry} → {template.upper().replace('_', ' ')}")

    # Manual template selection (used as fallback)
    template_options = {
        "healthtech": "🏥 HealthTech Template",
        "edtech": "📚 EdTech Template",
        "real_estate": "🏢 Real Estate / PropTech Template",
        "ai": "🤖 AI / Technology Template",
        "customize": "✏️ Customize Your Own"
    }

    selected_template = st.selectbox(
        "Default/Fallback Email Template",
        options=list(template_options.keys()),
        format_func=lambda x: template_options[x],
        index=list(template_options.keys()).index(mail_config["selected_template"]),
        key=f"template_select_{tab_name}",
        help="Used when no industry match found"
    )

    mail_config["selected_template"] = selected_template

    # Get current template
    if selected_template == "customize":
        current_template = mail_config["custom_template"]
    else:
        current_template = mail_config["permanent_templates"].get(selected_template, EMAIL_TEMPLATES[selected_template])

    st.markdown("---")
    st.markdown("#### 👀 Template Preview & Edit")

    # Show template
    if mail_config["template_edit_mode"]:
        st.info("✏️ **Edit Mode Active**")

        edited_template = st.text_area(
            "Edit Template",
            value=mail_config.get("temp_template", current_template),
            height=300,
            key=f"template_editor_{tab_name}"
        )

        mail_config["temp_template"] = edited_template

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("❌ Cancel", width='stretch', key=f"cancel_edit_{tab_name}"):
                mail_config["template_edit_mode"] = False
                mail_config["temp_template"] = ""
                st.rerun()

        with col2:
            if st.button("💾 Save (Session)", width='stretch', key=f"save_session_{tab_name}"):
                if selected_template == "customize":
                    mail_config["custom_template"] = edited_template
                else:
                    mail_config["permanent_templates"][selected_template] = edited_template
                mail_config["template_edit_mode"] = False
                st.success("✅ Template saved!")
                st.rerun()

        with col3:
            if st.button("🔒 Permanent Save", width='stretch', key=f"save_permanent_{tab_name}"):
                EMAIL_TEMPLATES[selected_template] = edited_template
                mail_config["permanent_templates"][selected_template] = edited_template
                mail_config["template_edit_mode"] = False
                st.success("✅ Template permanently saved!")
                st.rerun()

    else:
        st.code(current_template, language="text")

        if st.button("✏️ Edit Template", width='stretch', key=f"enable_edit_{tab_name}"):
            mail_config["template_edit_mode"] = True
            mail_config["temp_template"] = current_template
            st.rerun()

    st.markdown("---")

    # Send Emails Section
    st.markdown("#### 🚀 Send Emails")

    # Validation
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        valid_emails = sum(1 for email in mail_config["dummy_mail_addresses"] if email.strip())
        st.metric("Sender Accounts", f"{valid_emails}/{num_mails}")

    with col2:
        valid_oauth = sum(1 for cred in mail_config["oauth_credentials"] if cred is not None)
        st.metric("OAuth Loaded", f"{valid_oauth}/{num_mails}")

    with col3:
        if has_recipients and recipients_df is not None:
            st.metric("Recipients", len(recipients_df))
        else:
            st.metric("Recipients", "0")

    with col4:
        ready_to_send = valid_emails > 0 and valid_oauth > 0 and has_recipients
        status = "✅ Ready" if ready_to_send else "❌ Not Ready"
        st.metric("Status", status)

    # SINGLE Send Emails button - sets flag to show preview
    if st.button("📤 Check Emails Status ", type="primary", width='stretch', key=f"send_emails_{tab_name}"):
        errors = []
    
        if valid_emails == 0:
            errors.append("❌ Please provide at least one sender email address")
    
        if valid_oauth == 0:
            errors.append("❌ Please upload at least one OAuth credentials file")
    
        if not has_recipients:
            errors.append("❌ No recipient data found")
    
        if errors:
            for error in errors:
                st.error(error)
        else:
            # Store the intent to show preview in session state
            st.session_state[f"show_email_preview_{tab_name}"] = True
            st.rerun()

    # Show preview and confirmation section OUTSIDE the button (based on session state)
    if st.session_state.get(f"show_email_preview_{tab_name}", False) and has_recipients and recipients_df is not None:
        st.markdown("---")
        st.markdown("### 📧 Email Sending Preview")
    
        st.write("**Preview of emails to be sent:**")
        st.write("")
    
        email_preview_list = []
    
        for idx, row in recipients_df.iterrows():
            sender_idx = idx % valid_emails
            valid_senders = [(i, email) for i, email in enumerate(mail_config["dummy_mail_addresses"]) 
                           if email.strip() and mail_config["oauth_credentials"][i] is not None]
            
            if valid_senders:
                _, sender_email = valid_senders[idx % len(valid_senders)]
            else:
                sender_email = "No sender"
                
            recipient_email = row['Email']
            founder_name = row.get('Founder_Name', 'Recipient')
    
            # Auto-select template based on industry
            if 'Industry_Sector' in recipients_df.columns:
                selected_tmpl = map_industry_to_template(row['Industry_Sector'])
            else:
                selected_tmpl = mail_config["selected_template"]
    
            # Get template
            if selected_tmpl == "customize":
                template_text = mail_config["custom_template"]
            else:
                template_text = mail_config["permanent_templates"].get(selected_tmpl, EMAIL_TEMPLATES[selected_tmpl])
    
            # Replace placeholders
            personalized_email = safe_personalize_email(
                template_text,
                founder_name,
                row.get('Company_Name', 'Your Company') if 'Company_Name' in recipients_df.columns else 'Your Company',
                "Your Team"
            )
    
            email_preview_list.append({
                "Recipient": founder_name,
                "Email": recipient_email,
                "Sender": sender_email,
                "Industry": row.get('Industry_Sector', 'N/A') if 'Industry_Sector' in recipients_df.columns else 'N/A',
                "Template": selected_tmpl,
                "Preview": personalized_email[:100] + "..."
            })
    
        # Show preview table
        preview_df = pd.DataFrame(email_preview_list)
        st.dataframe(preview_df, width='stretch')
    
        # Confirm and send buttons
        st.markdown("---")
        
        col_confirm, col_cancel = st.columns(2)
        
        with col_confirm:
            confirm_send = st.button(
                "✅ Confirm & Send All Emails", 
                type="primary", 
                width='stretch', 
                key=f"confirm_send_{tab_name}"
            )
        
        with col_cancel:
            cancel_send = st.button(
                "❌ Cancel", 
                width='stretch', 
                key=f"cancel_send_{tab_name}"
            )
        
        if cancel_send:
            st.session_state[f"show_email_preview_{tab_name}"] = False
            st.rerun()
        
        if confirm_send:
            # Hide the preview for next run
            st.session_state[f"show_email_preview_{tab_name}"] = False
            
            # Show sending progress
            st.markdown("---")
            st.markdown("### 📧 Sending Emails...")
            
            try:
                # Create progress tracking
                progress_bar = st.progress(0)
                status_placeholder = st.empty()
                
                # Actually send the emails with live progress
                sent_count = 0
                failed_count = 0
                errors_list = []
                
                total_recipients = len(recipients_df)
                
                for idx, row in recipients_df.iterrows():
                    try:
                        # Update progress
                        progress_percent = (idx + 1) / total_recipients
                        progress_bar.progress(progress_percent)
                        status_placeholder.text(f"Sending {idx + 1}/{total_recipients}...")
                        
                        valid_senders = [(i, email) for i, email in enumerate(mail_config["dummy_mail_addresses"]) 
                                       if email.strip() and mail_config["oauth_credentials"][i] is not None]
                        
                        sender_idx_actual, sender_email = valid_senders[idx % len(valid_senders)]
                        oauth_creds = mail_config["oauth_credentials"][sender_idx_actual]
                        
                        recipient_email = row['Email']
                        founder_name = row.get('Founder_Name', 'Valued Recipient')
                        company_name = row.get('Company_Name', 'Unknown') if 'Company_Name' in recipients_df.columns else 'Unknown'
                        
                        # Determine template
                        if 'Industry_Sector' in recipients_df.columns:
                            selected_tmpl = map_industry_to_template(row['Industry_Sector'])
                        else:
                            selected_tmpl = mail_config["selected_template"]
                        
                        # Get template
                        if selected_tmpl == "customize":
                            template_text = mail_config["custom_template"]
                        else:
                            template_text = mail_config["permanent_templates"].get(
                                selected_tmpl, 
                                EMAIL_TEMPLATES[selected_tmpl]
                            )
                        
                        # Personalize email
                        personalized_email = safe_personalize_email(
                            template_text,
                            founder_name,
                            company_name,
                            "Your Team"
                        )
                        
                        # Extract subject and body
                        lines = personalized_email.split('\n')
                        subject = lines[0].replace('Subject:', '').strip() if lines and 'Subject' in lines[0] else "Hello"
                        body = '\n'.join(lines[1:]).strip() if len(lines) > 1 else personalized_email
                        
                        # ✅ FIXED: Create lead record FIRST to get lead_id
                        conn = sqlite3.connect("leads.db")
                        cursor = conn.cursor()
                        
                        cursor.execute("""
                            INSERT INTO leads (email, founder_name, company_name, template_used, sender_email, bounced)
                            VALUES (?, ?, ?, ?, ?, ?)
                        """, (recipient_email, founder_name, company_name, selected_tmpl, sender_email, 0))
                        
                        lead_id = cursor.lastrowid  # Get the auto-generated ID
                        conn.commit()
                        conn.close()
                        
                        log_to_debug(f"📝 Created lead record with ID: {lead_id}")
                        
                        # ✅ THEN: Send email with tracking pixel using lead_id
                        success, message, thread_id = send_email_via_gmail(
                            sender_email=sender_email,
                            oauth_credentials=oauth_creds,
                            recipient_email=recipient_email,
                            subject=subject,
                            body=body,
                            lead_id=lead_id  # ✅ Pass lead_id for tracking
                        )
                        
                        if success:
                            sent_count += 1
                            
                            # Update the record with thread_id
                            conn = sqlite3.connect("leads.db")
                            cursor = conn.cursor()
                            cursor.execute("""
                                UPDATE leads 
                                SET thread_id = ?
                                WHERE id = ?
                            """, (thread_id if thread_id else None, lead_id))
                            conn.commit()
                            conn.close()
                            
                            log_to_debug(f"✅ [{sent_count}/{total_recipients}] Sent to {recipient_email} | Lead ID: {lead_id} | Thread: {thread_id}")
                        else:
                            failed_count += 1
                            
                            # Mark as bounced
                            conn = sqlite3.connect("leads.db")
                            cursor = conn.cursor()
                            cursor.execute("""
                                UPDATE leads 
                                SET bounced = 1
                                WHERE id = ?
                            """, (lead_id,))
                            conn.commit()
                            conn.close()
                            
                            errors_list.append({
                                'recipient': recipient_email,
                                'error': message
                            })
                            log_to_debug(f"❌ Failed to send to {recipient_email}: {message}")
                        
                        import time
                        time.sleep(1)  # Rate limit
                    
                    except Exception as e:
                        failed_count += 1
                        errors_list.append({
                            'recipient': row.get('Email', 'Unknown'),
                            'error': str(e)
                        })
                        log_to_debug(f"❌ Error sending to {row.get('Email', 'Unknown')}: {str(e)}")
                
                # Clear progress indicators
                progress_bar.empty()
                status_placeholder.empty()
                
                # Show final results
                st.markdown("---")
                st.markdown("### 📊 Sending Results")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("✅ Sent", sent_count)
                with col2:
                    st.metric("❌ Failed", failed_count)
                with col3:
                    success_rate = (sent_count / total_recipients * 100) if total_recipients > 0 else 0
                    st.metric("Success Rate", f"{success_rate:.1f}%")
                
                if sent_count > 0:
                    st.success(f"✅ Successfully sent {sent_count} emails!")
                
                if errors_list:
                    with st.expander(f"⚠️ Failed to send ({len(errors_list)})"):
                        failed_df = pd.DataFrame(errors_list)
                        st.dataframe(failed_df, width='stretch')
                
                log_to_debug(f"Email sending complete: {sent_count} sent, {failed_count} failed")
            
            except Exception as e:
                st.error(f"❌ Critical Error: {str(e)}")
                log_to_debug(f"❌ Critical error during email sending: {str(e)}")
                
                import traceback
                with st.expander("🔍 Full Error Traceback"):
                    st.code(traceback.format_exc())
if 'extracted_info' not in st.session_state:
    st.session_state.extracted_info = None
if 'output_logs' not in st.session_state:
    st.session_state.output_logs = None
if 'df' not in st.session_state:
    st.session_state.df = None
if 'csv_filename' not in st.session_state:
    st.session_state.csv_filename = None
if 'enriched_df' not in st.session_state:
    st.session_state.enriched_df = None
if 'dedup_result' not in st.session_state:
    st.session_state.dedup_result = None
if 'dedup_stats' not in st.session_state:
    st.session_state.dedup_stats = None
if "llm_config" not in st.session_state:
    st.session_state.llm_config = {
        "provider": "Mistral",
        "api_key": "",
        "api_key_2": "",
        "validated": False,
        "validated_2": False,
        "custom_endpoint": "",
        "custom_endpoint_2": ""
    }
if "debug_logs" not in st.session_state:
    st.session_state.debug_logs = []
if "show_endpoint_editor" not in st.session_state:
    st.session_state.show_endpoint_editor = {}
if "temp_endpoints" not in st.session_state:
    st.session_state.temp_endpoints = {}
if "selected_llm" not in st.session_state:
    st.session_state.selected_llm = None


# ============ SIDEBAR CONFIGURATION ============

st.sidebar.title("⚙️ LLM Configuration")

st.sidebar.info("💡 **Tip:** Provide 2 API keys for the same provider for better reliability and load distribution.")

config = st.session_state.llm_config

with st.sidebar.expander(f"🔧 {config['provider']} Configuration", expanded=True):
    previous_provider = config.get("provider", "Mistral")
    provider = st.selectbox(
        "Select LLM Provider",
        ["Claude", "OpenAI", "Gemini", "Mistral", "Deepseek", "Qwen", "Perplexity", "Llama"],
        key="provider_select",
        index=["Claude", "OpenAI", "Gemini", "Mistral", "Deepseek", "Qwen", "Perplexity", "Llama"].index(config["provider"])
    )
    
    if provider != previous_provider:
        st.session_state.llm_config["custom_endpoint"] = ""
        st.session_state.llm_config["custom_endpoint_2"] = ""
        st.session_state.llm_config["validated"] = False
        st.session_state.llm_config["validated_2"] = False
        st.session_state["show_edit_btn_key1"] = False
        st.session_state["show_edit_btn_key2"] = False
        st.session_state["validation_error_key1"] = ""
        st.session_state["validation_error_key2"] = ""
        st.session_state.show_endpoint_editor["key1"] = False
        st.session_state.show_endpoint_editor["key2"] = False
        if "key1" in st.session_state.temp_endpoints:
            del st.session_state.temp_endpoints["key1"]
        if "key2" in st.session_state.temp_endpoints:
            del st.session_state.temp_endpoints["key2"]
        log_to_debug(f"🔄 Provider changed from {previous_provider} to {provider} - endpoints reset to default")
    
    st.session_state.llm_config["provider"] = provider
    
    # First API Key
    api_key = st.text_input(
        f"{provider} API Key",
        value=config.get("api_key", ""),
        type="password",
        key="api_key_input"
    )
    st.session_state.llm_config["api_key"] = api_key
    
    if st.session_state.show_endpoint_editor.get("key1", False):
        if "key1" not in st.session_state.temp_endpoints:
            st.session_state.temp_endpoints["key1"] = get_active_endpoint(config, 1)
        
        st.caption(f"🔗 Current Endpoint:")
        edited_endpoint = st.text_input(
            "Edit Endpoint URL",
            value=st.session_state.temp_endpoints["key1"],
            key="endpoint_edit_key1",
            help="Modify this endpoint if needed. Changes will be saved when you click Save."
        )
        st.session_state.temp_endpoints["key1"] = edited_endpoint
        
        col_save, col_cancel = st.columns(2)
        with col_save:
            if st.button("💾 Save", key="save_endpoint_key1", width='stretch'):
                st.session_state.llm_config["custom_endpoint"] = edited_endpoint
                log_to_debug(f"📝 Endpoint updated for {provider}: {edited_endpoint}")
                st.session_state.show_endpoint_editor["key1"] = False
                st.session_state["show_edit_btn_key1"] = False
                st.success("✅ Endpoint saved! Please validate again.")
                st.rerun()
        
        with col_cancel:
            if st.button("❌ Cancel", key="cancel_endpoint_key1", width='stretch'):
                st.session_state.show_endpoint_editor["key1"] = False
                st.session_state["show_edit_btn_key1"] = False
                if "key1" in st.session_state.temp_endpoints:
                    del st.session_state.temp_endpoints["key1"]
                st.rerun()
    
    if api_key and not st.session_state.show_endpoint_editor.get("key1", False):
        show_edit_btn = st.session_state.get("show_edit_btn_key1", False)
        
        if show_edit_btn and st.session_state.get("validation_error_key1"):
            st.error(st.session_state["validation_error_key1"])
        
        if st.button(f"✓ Validate API Key", key="validate_key1", width='stretch'):
            endpoint = get_active_endpoint(config, 1)
            is_valid, message = test_llm_connection(provider, api_key, endpoint)
            
            if is_valid:
                st.session_state.llm_config["validated"] = True
                st.session_state["show_edit_btn_key1"] = False
                st.session_state["validation_error_key1"] = ""
                log_to_debug(f"✅ {provider} API Key is Valid (Endpoint: {endpoint})")
                st.success(f"✅ API key is Valid")
                st.rerun()
            else:
                st.session_state.llm_config["validated"] = False
                st.session_state["show_edit_btn_key1"] = True
                st.session_state["validation_error_key1"] = f"❌ Authentication failed: {message}"
                log_to_debug(f"❌ {provider} API Key validation failed at {endpoint}: {message}")
                st.rerun()
        
        if show_edit_btn:
            if st.button("🔧 Edit Endpoint", key="edit_endpoint_btn_key1", width='stretch'):
                st.session_state.show_endpoint_editor["key1"] = True
                st.rerun()
        
        if config.get("validated"):
            active_endpoint = get_active_endpoint(config, 1)
            st.success(f"✅ API key is Valid")
            st.caption(f"🔗 Using: `{active_endpoint}`")
    
    # Second API Key
    api_key_2 = st.text_input(
        f"{provider} API Key (Optional)",
        value=config.get("api_key_2", ""),
        type="password",
        key="api_key_2_input",
        help="Recommended: Add a second key for automatic failover and load distribution"
    )
    st.session_state.llm_config["api_key_2"] = api_key_2
    
    if st.session_state.show_endpoint_editor.get("key2", False):
        if "key2" not in st.session_state.temp_endpoints:
            st.session_state.temp_endpoints["key2"] = get_active_endpoint(config, 2)
        
        st.caption(f"🔗 Current Endpoint (Key 2):")
        edited_endpoint_2 = st.text_input(
            "Edit Endpoint URL",
            value=st.session_state.temp_endpoints["key2"],
            key="endpoint_edit_key2",
            help="Modify this endpoint if needed. Changes will be saved when you click Save."
        )
        st.session_state.temp_endpoints["key2"] = edited_endpoint_2
        
        col_save2, col_cancel2 = st.columns(2)
        with col_save2:
            if st.button("💾 Save", key="save_endpoint_key2", width='stretch'):
                st.session_state.llm_config["custom_endpoint_2"] = edited_endpoint_2
                log_to_debug(f"📝 Endpoint updated for {provider} (Key 2): {edited_endpoint_2}")
                st.session_state.show_endpoint_editor["key2"] = False
                st.session_state["show_edit_btn_key2"] = False
                st.success("✅ Endpoint saved! Please validate again.")
                st.rerun()
        
        with col_cancel2:
            if st.button("❌ Cancel", key="cancel_endpoint_key2", width='stretch'):
                st.session_state.show_endpoint_editor["key2"] = False
                st.session_state["show_edit_btn_key2"] = False
                if "key2" in st.session_state.temp_endpoints:
                    del st.session_state.temp_endpoints["key2"]
                st.rerun()
    
    if api_key_2 and not st.session_state.show_endpoint_editor.get("key2", False):
        show_edit_btn_2 = st.session_state.get("show_edit_btn_key2", False)
        
        if show_edit_btn_2 and st.session_state.get("validation_error_key2"):
            st.error(st.session_state["validation_error_key2"])
        
        if st.button(f"✓ Validate API Key", key="validate_key2", width='stretch'):
            endpoint = get_active_endpoint(config, 2)
            is_valid, message = test_llm_connection(provider, api_key_2, endpoint)
            
            if is_valid:
                st.session_state.llm_config["validated_2"] = True
                st.session_state["show_edit_btn_key2"] = False
                st.session_state["validation_error_key2"] = ""
                log_to_debug(f"✅ {provider} second API Key validated (Endpoint: {endpoint})")
                st.success(f"✅ API key authenticated ")
                st.rerun()
            else:
                st.session_state.llm_config["validated_2"] = False
                st.session_state["show_edit_btn_key2"] = True
                st.session_state["validation_error_key2"] = f"❌ Authentication failed: {message}"
                log_to_debug(f"❌ {provider} second API Key validation failed at {endpoint}: {message}")
                st.rerun()
        
        if show_edit_btn_2:
            if st.button("🔧 Edit Endpoint", key="edit_endpoint_btn_key2", width='stretch'):
                st.session_state.show_endpoint_editor["key2"] = True
                st.rerun()
        
        if config.get("validated_2"):
            active_endpoint = get_active_endpoint(config, 2)
            st.success(f"✅ API Key is Valid ")
            st.caption(f"🔗 Using: `{active_endpoint}`")
# Add this to your SIDEBAR CONFIGURATION section

st.sidebar.markdown("---")
st.sidebar.title("🌐 Tracking Server")

# Initialize server state
if 'tracking_server_running' not in st.session_state:
    st.session_state.tracking_server_running = False
if 'tracking_server_url' not in st.session_state:
    st.session_state.tracking_server_url = None
if 'cloudflare_process' not in st.session_state:
    st.session_state.cloudflare_process = None

# Display current status
if not st.session_state.tracking_server_running:
    st.sidebar.error("🔴 Server is NOT running")
    
    if st.sidebar.button("🚀 Start Tracking Server", use_container_width=True):
        with st.spinner("⏳ Starting server... This takes 30-40 seconds"):
            try:
                # Import the function
                from terminal_automated import get_tunnel_url
                
                # Get the tunnel URL
                tunnel_url, flask_thread, cloudflare_process = get_tunnel_url()
                
                # Update session state
                st.session_state.tracking_server_url = tunnel_url
                st.session_state.tracking_server_running = True
                st.session_state.cloudflare_process = cloudflare_process
                
                # Update the global variable
                globals()['TRACKING_SERVER_URL'] = tunnel_url
                
                log_to_debug(f"✅ Tracking server started: {tunnel_url}")
                st.sidebar.success(f"✅ Server started!")
                st.rerun()
                
            except ImportError:
                st.sidebar.error("❌ terminal_automated.py not found!")
                log_to_debug("❌ terminal_automated.py not found")
            except Exception as e:
                st.sidebar.error(f"❌ Failed to start: {str(e)}")
                log_to_debug(f"❌ Server start error: {str(e)}")
                import traceback
                with st.sidebar.expander("🔍 Error Details"):
                    st.code(traceback.format_exc())

else:
    # Server is running
    st.sidebar.success("🟢 Server is RUNNING")
    st.sidebar.info(f"🌐 URL: {st.session_state.tracking_server_url}")
    st.sidebar.caption("Copy this URL to use in email tracking")
    
    # Copy button (shows the URL in a text box)
    st.sidebar.code(st.session_state.tracking_server_url, language=None)
    
    if st.sidebar.button("🛑 Stop Server", use_container_width=True):
        try:
            if st.session_state.cloudflare_process:
                st.session_state.cloudflare_process.terminate()
                st.session_state.cloudflare_process = None
            
            st.session_state.tracking_server_running = False
            st.session_state.tracking_server_url = None
            
            log_to_debug("🛑 Tracking server stopped")
            st.sidebar.success("✅ Server stopped")
            st.rerun()
            
        except Exception as e:
            st.sidebar.error(f"❌ Error stopping: {str(e)}")
            log_to_debug(f"❌ Server stop error: {str(e)}")

st.sidebar.caption("⚠️ Keep this running while sending emails")

st.sidebar.divider()
st.sidebar.caption("Lead Generation Agent")

# ============ MAIN CONTENT ============

tab1, tab2, tab3, tab4 = st.tabs(["📊 Venture Capital & IPO", "🔗 LinkedIn Scraper", "🔄 Deduplicator", "📈 Dashboard"])

# ============ TAB 1: LEAD GENERATION ============

# ============ TAB 1: LEAD GENERATION ============
# ============ TAB 1: LEAD GENERATION ============
with tab1:
    st.title("📊 Venture Capital & IPO")
    st.markdown("Select your preferences to generate a funding report!")

    st.markdown("### 🔍 Select Report Parameters")

    # ============ NEW: VC/IPO Selection Dropdown ============
    col_type = st.columns(1)[0]
    
    with col_type:
        report_type = st.selectbox(
            "📋 Report Type",
            options=["Venture Capital", "IPO"],
            key="report_type_select",
            help="Select the type of report you want to generate"
        )
    
    # Determine if IPO is selected
    is_ipo_mode = (report_type == "IPO")
    
    st.markdown("---")
    
    # ============ CONDITIONAL RENDERING BASED ON REPORT TYPE ============
    
    if is_ipo_mode:
        # ✅ IPO MODE - Show disabled/grayed out dropdowns
        st.info("🏢 **IPO Mode Selected** - Timeframe, Region, and Sector options are not applicable for IPO reports")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.selectbox(
                "📅 Timeframe",
                options=["Not Applicable for IPO"],
                key="timeframe_select_disabled",
                disabled=True,
                help="Timeframe selection is disabled in IPO mode"
            )
        
        with col2:
            st.selectbox(
                "🌍 Region",
                options=["Not Applicable for IPO"],
                key="region_select_disabled",
                disabled=True,
                help="Region selection is disabled in IPO mode"
            )
        
        with col3:
            st.selectbox(
                "🏢 Sector",
                options=["Not Applicable for IPO"],
                key="sector_select_disabled",
                disabled=True,
                help="Sector selection is disabled in IPO mode"
            )
        
        auto_query = "IPO Companies Report"
        
    else:
        # ✅ VENTURE CAPITALIST MODE - Show normal active dropdowns
        col1, col2, col3 = st.columns(3)

        with col1:
            today = datetime.now()
            current_weekday = today.weekday()
            days_since_monday = current_weekday + 7
            last_monday = today - timedelta(days=days_since_monday)
            last_sunday = last_monday + timedelta(days=6)
            last_week = f"{last_monday.strftime('%d %B')} - {last_sunday.strftime('%d %B %Y')}"
            
            first_day_this_month = today.replace(day=1)
            last_month = (first_day_this_month - timedelta(days=1)).strftime("%B %Y")
            
            timeframe = st.selectbox(
                "📅 Timeframe",
                options=[
                    f" Weekly Report  ({last_week})",
                    f" Monthly Report  ({last_month})",
                ],
                key="timeframe_select"
            )

        with col2:
            region = st.selectbox(
                "🌍 Region",
                options=["Global", "India"],
                help="Select the geographical region",
                key="region_select"
            )

        with col3:
            sector_options = get_sector_options(timeframe, region)
            sector = st.selectbox(
                "🏢 Sector",
                options=sector_options,
                help="Select the industry sector",
                key="sector_select"
            )

        auto_query = f"{sector} startups funding in {region} {timeframe}"

    st.markdown("---")
    st.markdown("### 📝 Generated Query")
    st.info(f"**Query:** {auto_query}")

    st.markdown("---")

    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        extract_button = st.button("🔍 Generate Report", type="primary", width='stretch')
    with col2:
        clear_button = st.button("🗑️ Clear History", width='stretch')

    if clear_button:
        clear_history()
        st.rerun()

    if extract_button:
        config = st.session_state.llm_config
        has_valid_config = config.get("api_key") and config.get("validated")
        
        if not has_valid_config:
            st.error("❌ Please configure and validate at least one LLM API key in the sidebar before generating reports")
            log_to_debug("❌ Generate Report failed: No validated LLM configuration found")
            st.info("💡 Go to the sidebar 'LLM Configuration' section to add and validate your API key")
        else:
            with st.spinner("Processing your request... ( this may take 20-25 minutes ) "):
                if is_ipo_mode:
                    # ============ IPO MODE ============
                    log_to_debug("🏢 IPO mode selected - calling scrape_zerodha_ipos")
                    
                    extracted_info = {
                        'report_type': 'IPO',
                        'sector': 'IPO',
                        'region': 'N/A',
                        'timeframe': 'N/A'
                    }
                    st.session_state.extracted_info = extracted_info
    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
    
                    try:
                        status_text.text("🏢 Scraping IPO data from Zerodha...")
                        progress_bar.progress(20)

                        primary_key = config.get("api_key", "")
                        secondary_key = config.get("api_key_2", "")
                        
                        selected_llm = {
                            "provider": config["provider"],
                            "api_key": primary_key,
                            "endpoint": get_active_endpoint(config, 1),
                            "validated": True
                        }
                        
                        st.session_state.selected_llm = selected_llm
                        
                        log_to_debug(f"🔍 Using LLM provider for IPO: {selected_llm.get('provider')}")
                        log_to_debug(f"🔑 Primary key: {primary_key[:10] if primary_key else 'EMPTY'}...")
                        log_to_debug(f"🔑 Secondary key: {secondary_key[:10] if secondary_key else 'None'}...")

                        # ✅ CALL YOUR IPO FUNCTION HERE
                        enriched_df, output_file = scrape_zerodha_ipos(
                            llm_provider=config["provider"],
                            api_key_1=primary_key,
                            api_key_2=secondary_key if secondary_key else None,
                            output_csv="zerodha_ipos_enriched.xlsx"
                        )

                        progress_bar.progress(80)
                        status_text.text("✅ IPO data enriched!")

                        # Store results in session state
                        st.session_state.df = enriched_df
                        st.session_state.csv_filename = output_file

                        progress_bar.progress(100)
                        status_text.text("✅ IPO report generated!")

                        if output_file and os.path.exists(output_file):
                            st.success(f"✅ Successfully loaded IPO report: {output_file} ({len(enriched_df)} rows)")
                            log_to_debug(f"✅ IPO report loaded: {len(enriched_df)} companies")
                        else:
                            st.error("❌ No result file returned from IPO script")
                            log_to_debug("❌ IPO script did not return a valid file")
    
                    except Exception as e:
                        st.error(f"❌ Error during IPO report generation: {str(e)}")
                        log_to_debug(f"❌ IPO generation error: {str(e)}")
                        import traceback
                        with st.expander("🔍 Error Details"):
                            st.code(traceback.format_exc())
                
                else:
                    # ============ VENTURE CAPITALIST MODE ============
                    log_to_debug("💼 Venture Capitalist mode selected - calling main.py")
                    
                    extracted_info = {
                        'report_type': 'Venture Capitalist',
                        'sector': sector,
                        'region': region,
                        'timeframe': timeframe
                    }
                    st.session_state.extracted_info = extracted_info
    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                
                    try:
                        from main import start

                        status_text.text("🔍 Analyzing intent...")
                        progress_bar.progress(10)

                        import sys
                        from io import StringIO

                        old_stdout = sys.stdout
                        sys.stdout = captured_output = StringIO()

                        status_text.text("🔍 Searching for companies and enriching data...")
                        progress_bar.progress(30)

                        primary_key = config.get("api_key", "")
                        secondary_key = config.get("api_key_2", "")
                        
                        selected_llm = {
                            "provider": config["provider"],
                            "api_key": primary_key,
                            "endpoint": get_active_endpoint(config, 1),
                            "validated": True
                        }
                        
                        st.session_state.selected_llm = selected_llm
                        
                        log_to_debug(f"🔍 Using LLM provider: {selected_llm.get('provider')}")
                        log_to_debug(f"🔑 Primary key: {primary_key[:10] if primary_key else 'EMPTY'}...")
                        log_to_debug(f"🔑 Secondary key: {secondary_key[:10] if secondary_key else 'None'}...")

                        csv_filename = start(
                            sector=sector,
                            region=region,
                            timeframe=timeframe,
                            mistral_key_1=primary_key,
                            mistral_key_2=secondary_key,
                            llm_config=selected_llm
                        )

                        sys.stdout = old_stdout
                        output = captured_output.getvalue()

                        st.session_state.output_logs = output
                        st.session_state.csv_filename = csv_filename

                        progress_bar.progress(100)
                        status_text.text("✅ Processing complete!")

                        if csv_filename:
                            with st.expander("🔍 Debug Information", expanded=False):
                                st.write("CSV/Excel file returned:", csv_filename)
                                st.write("File exists:", os.path.exists(csv_filename) if csv_filename else False)

                                if csv_filename and os.path.exists(csv_filename):
                                    try:
                                        if csv_filename.endswith('.xlsx') or csv_filename.endswith('.xls'):
                                            df = pd.read_excel(csv_filename)
                                            st.info(f"📊 Loaded Excel file: {csv_filename}")
                                        else:
                                            df = pd.read_csv(csv_filename)
                                            st.info(f"📊 Loaded CSV file: {csv_filename}")
                                        
                                        st.session_state.df = df
                                        st.session_state.csv_filename = csv_filename
                                        st.success(f"✅ Successfully loaded: {csv_filename} ({len(df)} rows)")
                                    except Exception as e:
                                        st.error(f"❌ Error loading file: {str(e)}")
                                        import traceback
                                        with st.expander("🔍 Error Details"):
                                            st.code(traceback.format_exc())
                                elif csv_filename:
                                    st.warning(f"⚠️ File not found: {csv_filename}")
                                    with st.expander("🔍 Available files in directory"):
                                        files = [f for f in os.listdir('.') if f.endswith(('.csv', '.xlsx', '.xls'))]
                                        st.write("Files found:", files)
                                else:
                                    st.warning("⚠️ No filename returned from processing")
                                    all_files = [f for f in os.listdir('.') if f.endswith(('.csv', '.xlsx', '.xls'))]
                                    if all_files:
                                        all_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                                        csv_filename = all_files[0]
                                        st.info(f"🔍 Found most recent file: {csv_filename}")
                                        try:
                                            if csv_filename.endswith('.xlsx') or csv_filename.endswith('.xls'):
                                                df = pd.read_excel(csv_filename)
                                            else:
                                                df = pd.read_csv(csv_filename)
                                            st.session_state.df = df
                                            st.session_state.csv_filename = csv_filename
                                            st.success(f"✅ Loaded file: {csv_filename} ({len(df)} rows)")
                                        except Exception as e:
                                            st.error(f"❌ Error loading file: {str(e)}")
                                    else:
                                        st.error("❌ No CSV/Excel files found in directory")
                        else:
                            st.error("❌ No result returned from processing")

                    except Exception as e:
                        sys.stdout = old_stdout
                        st.error(f"❌ Error during processing: {str(e)}")
                        import traceback
                        with st.expander("🔍 Error Details"):
                            st.code(traceback.format_exc())

    if st.session_state.extracted_info:
        st.success("✅ Information extracted successfully!")

        report_type_val = st.session_state.extracted_info.get('report_type', 'Venture Capitalist')
        sector_val = st.session_state.extracted_info.get('sector', 'Not specified')
        region_val = st.session_state.extracted_info.get('region', 'Not specified')
        timeframe_val = st.session_state.extracted_info.get('timeframe', 'Not specified')

        if report_type_val == 'IPO':
            # IPO Mode - Show only report type
            col_single = st.columns(1)[0]
            with col_single:
                st.metric(
                    label="📋 Report Type",
                    value="IPO Companies"
                )
        else:
            # VC Mode - Show all metrics
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    label="📅 Timeframe",
                    value=timeframe_val
                )

            with col2:
                st.metric(
                    label="🌍 Region",
                    value=region_val
                )

            with col3:
                st.metric(
                    label="🏢 Sector",
                    value=sector_val
                )

        st.markdown("---")

        if st.session_state.output_logs:
            log_expander = st.expander("📋 View Detailed Logs", expanded=False)
            with log_expander:
                st.code(st.session_state.output_logs, language="text")

        if st.session_state.df is not None:
            st.markdown("---")
            st.markdown("### 📄 Generated Report")

            df = st.session_state.df

            st.success(f"✅ Report generated successfully! ({len(df)} companies)")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Companies", len(df))
            
            with col2:
                email_col = None
                for col in df.columns:
                    if col.lower() == 'email':
                        email_col = col
                        break
                email_count = df[email_col].notna().sum() if email_col else 0
                st.metric("With Email", email_count)
            
            with col3:
                phone_col = None
                for col in df.columns:
                    if col.lower() == 'phone':
                        phone_col = col
                        break
                phone_count = df[phone_col].notna().sum() if phone_col else 0
                st.metric("With Phone", phone_count)
            
            with col4:
                linkedin_col = None
                for col in df.columns:
                    if 'linkedin' in col.lower():
                        linkedin_col = col
                        break
                linkedin_count = df[linkedin_col].notna().sum() if linkedin_col else 0
                st.metric("With LinkedIn", linkedin_count)

            st.markdown("#### 📊 Data Preview")
            st.dataframe(
                df,
                width='stretch',
                height=400
            )

            csv_data = df.to_csv(index=False).encode('utf-8')
            
            download_filename = st.session_state.csv_filename
            if download_filename and download_filename.endswith('.xlsx'):
                download_filename = download_filename.replace('.xlsx', '.csv')
            
            st.download_button(
                label="⬇️ Download Full Report (CSV)",
                data=csv_data,
                file_name=download_filename if download_filename else "funding_report.csv",
                mime="text/csv",
                width='stretch',
                key="download_csv"
            )

            # ============ EXTRACT CONTACT SECTION ============
            st.markdown("---")
            st.markdown("### 📧 Extract Contact Information")
            st.caption("Enrich your data with contact information using Apollo.io API")
            
            # CSV Selection Options
            csv_selection = st.radio(
                "Select CSV file to enrich:",
                options=["Use Generated CSV", "Upload Local CSV"],
                horizontal=True,
                key="csv_selection_radio"
            )
            
            csv_to_enrich = None
            
            if csv_selection == "Use Generated CSV":
                if st.session_state.df is not None:
                    csv_to_enrich = st.session_state.df
                    st.info(f"📊 Using generated CSV with {len(csv_to_enrich)} rows")
                else:
                    st.warning("⚠️ No generated CSV available. Please generate a report first or upload a local CSV.")
            
            else:  # Upload Local CSV
                uploaded_csv = st.file_uploader(
                    "Upload CSV file",
                    type=["csv"],
                    key="upload_csv_for_extraction"
                )
                
                if uploaded_csv:
                    try:
                        csv_to_enrich = pd.read_csv(uploaded_csv)
                        st.info(f"📊 Uploaded CSV with {len(csv_to_enrich)} rows")
                        
                        # Show preview of uploaded CSV
                        with st.expander("📋 Preview Uploaded CSV", expanded=False):
                            st.dataframe(csv_to_enrich.head(10), width='stretch')
                    
                    except Exception as e:
                        st.error(f"❌ Error reading CSV file: {str(e)}")
                        csv_to_enrich = None
            
            # Apollo API Key Input
            apollo_api_key = st.text_input(
                "🔑 Apollo.io API Key",
                type="password",
                key="apollo_api_key_input",
                help="Enter your Apollo.io API key to enrich contact data"
            )
            
            # Extract Emails Button
            if st.button("🚀 Extract Contact Info", type="primary", width='stretch', key="extract_contact_btn"):
                if not apollo_api_key:
                    st.error("❌ Please provide Apollo.io API key")
                    log_to_debug("❌ Email extraction failed: No Apollo API key provided")
                
                elif csv_to_enrich is None:
                    st.error("❌ Please select or upload a CSV file")
                    log_to_debug("❌ Email extraction failed: No CSV file selected")
                
                else:
                    log_to_debug(f"🚀 Starting email extraction for {len(csv_to_enrich)} records")
                    
                    with st.spinner(f"Extracting contact information for {len(csv_to_enrich)} companies... This may take a few minutes"):
                        try:
                            # Call the extraction function
                            enriched_df = extract_with_apollo(csv_to_enrich, apollo_api_key)
                            
                            # Store enriched dataframe in session state
                            st.session_state.enriched_df = enriched_df
                            
                            log_to_debug(f"✅ Email extraction completed successfully")
                            st.success(f"✅ Contact extraction completed! Processed {len(enriched_df)} records")
                        
                        except Exception as e:
                            st.error(f"❌ Error during contact extraction: {str(e)}")
                            log_to_debug(f"❌ Email extraction error: {str(e)}")
                            import traceback
                            with st.expander("🔍 Error Details"):
                                st.code(traceback.format_exc())
            
            # Display Enriched Data Preview
            if st.session_state.enriched_df is not None:
                st.markdown("---")
                st.markdown("### 📊 Enriched Data Preview")
                
                enriched_df = st.session_state.enriched_df
                
                # Show metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Records", len(enriched_df))
                
                with col2:
                    email_col = None
                    for col in enriched_df.columns:
                        if col.lower() == 'email':
                            email_col = col
                            break
                    email_count = enriched_df[email_col].notna().sum() if email_col else 0
                    st.metric("Emails Found", email_count)
                
                with col3:
                    phone_col = None
                    for col in enriched_df.columns:
                        if col.lower() == 'phone':
                            phone_col = col
                            break
                    phone_count = enriched_df[phone_col].notna().sum() if phone_col else 0
                    st.metric("Phones Found", phone_count)
                
                with col4:
                    designation_col = None
                    for col in enriched_df.columns:
                        if col.lower() == 'designation':
                            designation_col = col
                            break
                    designation_count = enriched_df[designation_col].notna().sum() if designation_col else 0
                    st.metric("Designations Found", designation_count)
                
                # Show dataframe
                st.dataframe(
                    enriched_df,
                    width='stretch',
                    height=400
                )
                
                # Download button for enriched data
                enriched_csv_data = enriched_df.to_csv(index=False).encode('utf-8')
                
                st.download_button(
                    label="⬇️ Download Enriched Report (CSV)",
                    data=enriched_csv_data,
                    file_name="enriched_report.csv",
                    mime="text/csv",
                    width='stretch',
                    key="download_enriched_csv"
                )

            # ============ GOOGLE SHEETS INTEGRATION ============
            st.markdown("---")
            
    # Render Mail Sender Section for Tab 1
    render_mail_sender_section("(Tab 1)")

# ============ TAB 2: LINKEDIN SCRAPER ============
with tab2:
    st.title("🔗 LinkedIn Scraper")
    st.markdown("Scrape LinkedIn posts and enrich with contact information using AI")
   
    # Initialize LinkedIn scraper config in session state
    if "linkedin_scraper_config" not in st.session_state:
        st.session_state.linkedin_scraper_config = {
            "apollo_api_key": "",
            "enrich_emails": False,
            "enrich_industries": False,
            "linkedin_email": "",
            "linkedin_password": "",
            "scrape_urls": ["", "", "", "", ""],
            "posts_per_url": 20
        }
   
    if "linkedin_scraped_data" not in st.session_state:
        st.session_state.linkedin_scraped_data = None
   
    linkedin_config = st.session_state.linkedin_scraper_config
   
    # ============ SIDEBAR CONFIGURATION ============
    st.sidebar.markdown("---")
    

    st.sidebar.markdown("### 🔗 LinkedIn Scraper Config")
   
    with st.sidebar.expander("⚙️ API Configuration", expanded=True):
        # Use the main LLM config from Tab 1
        st.markdown("**Using LLM Configuration from Tab 1**")
        main_llm_config = st.session_state.llm_config
       
        if main_llm_config.get("validated") and main_llm_config.get("api_key"):
            st.success(f"✅ Using {main_llm_config['provider']} API key")
            st.caption(f"Provider: {main_llm_config['provider']}")
        else:
            st.warning("⚠️ Please configure and validate LLM API key in Tab 1")
       
        st.markdown("---")
       
        # Apollo API Key (Required)
        st.markdown("**Apollo.io API Key** (Required)")
        apollo_key = st.text_input(
            "Enter Apollo.io API Key",
            value=linkedin_config.get("apollo_api_key", ""),
            type="password",
            key="linkedin_apollo_key",
            help="Required for email enrichment"
        )
        linkedin_config["apollo_api_key"] = apollo_key
       
        st.markdown("---")
       
        # Enrichment Options
        st.markdown("**Enrichment Options**")
       
        enrich_emails = st.checkbox(
            "🔍 Enrich with Emails",
            value=linkedin_config.get("enrich_emails", False),
            key="enrich_emails_checkbox",
            help="Extract email addresses using Apollo.io"
        )
        linkedin_config["enrich_emails"] = enrich_emails
       
        enrich_industries = st.checkbox(
            "🏢 Enrich with Industry Types",
            value=linkedin_config.get("enrich_industries", False),
            key="enrich_industries_checkbox",
            help="Analyze company industry and category using Mistral AI"
        )
        linkedin_config["enrich_industries"] = enrich_industries
   
    # ============ MAIN CONTENT ============
    st.markdown("### 🔐 LinkedIn Credentials")
   
    col1, col2 = st.columns(2)
   
    with col1:
        linkedin_email = st.text_input(
            "LinkedIn Email",
            value=linkedin_config.get("linkedin_email", ""),
            key="linkedin_email_input",
            placeholder="your.email@example.com"
        )
        linkedin_config["linkedin_email"] = linkedin_email
   
    with col2:
        linkedin_password = st.text_input(
            "LinkedIn Password",
            value=linkedin_config.get("linkedin_password", ""),
            type="password",
            key="linkedin_password_input"
        )
        linkedin_config["linkedin_password"] = linkedin_password
   
    st.markdown("---")
    st.markdown("### 🔗 Scraping Configuration")
   
    # Posts per URL dropdown
    posts_per_url = st.selectbox(
        "📊 Posts per URL",
        options=[20, 40, 60],
        index=[20, 40, 60].index(linkedin_config.get("posts_per_url", 20)),
        key="posts_per_url_select",
        help="Number of posts to scrape from each URL"
    )
    linkedin_config["posts_per_url"] = posts_per_url
   
    st.markdown("### 🌐 LinkedIn Search URLs (Add 1-5 URLs)")
    st.caption("Enter LinkedIn search URLs to scrape posts from. You can add between 1 to 5 URLs.")
   
    # URL input fields (1-5)
    scrape_urls = []
    for i in range(5):
        url = st.text_input(
            f"Search URL {i+1}" + (" (Required)" if i == 0 else " (Optional)"),
            value=linkedin_config["scrape_urls"][i] if i < len(linkedin_config["scrape_urls"]) else "",
            key=f"scrape_url_{i+1}",
            placeholder="https://www.linkedin.com/search/results/content/?keywords=...",
            help=f"LinkedIn search URL #{i+1}"
        )
        if url.strip():
            scrape_urls.append(url.strip())
   
    linkedin_config["scrape_urls"] = scrape_urls
   
    # Display configuration summary
    if scrape_urls:
        st.markdown("---")
        st.markdown("### 📊 Scraping Summary")
       
        col1, col2, col3 = st.columns(3)
       
        with col1:
            st.metric("URLs to Scrape", len(scrape_urls))
       
        with col2:
            st.metric("Posts per URL", posts_per_url)
       
        with col3:
            estimated_total = len(scrape_urls) * posts_per_url
            st.metric("Estimated Total Posts", estimated_total)
   
    st.markdown("---")
   
    # Validation and Start button
    col1, col2, col3 = st.columns([1, 1, 2])
   
    with col1:
        start_scraping = st.button("🚀 Start Scraping", type="primary", width='stretch', key="start_linkedin_scraping")
   
    with col2:
        clear_scraper = st.button("🗑️ Clear Data", width='stretch', key="clear_linkedin_scraper")
   
    if clear_scraper:
        st.session_state.linkedin_scraped_data = None
        st.session_state.linkedin_scraper_config["scrape_urls"] = ["", "", "", "", ""]
        st.success("✅ LinkedIn scraper data cleared!")
        log_to_debug("LinkedIn scraper data cleared")
        st.rerun()
   
    # Validation and scraping logic
    if start_scraping:
        # Validate inputs
        errors = []
       
        # Get LLM config from Tab 1
        main_llm_config = st.session_state.llm_config
       
        if not main_llm_config.get("validated") or not main_llm_config.get("api_key"):
            errors.append("❌ Please configure and validate your LLM API key in Tab 1 (Lead Generation)")
       
        if not linkedin_config.get("apollo_api_key"):
            errors.append("❌ Please provide your Apollo.io API key in the sidebar")
       
        if not linkedin_config.get("linkedin_email"):
            errors.append("❌ LinkedIn email is required")
       
        if not linkedin_config.get("linkedin_password"):
            errors.append("❌ LinkedIn password is required")
       
        if not scrape_urls or len(scrape_urls) == 0:
            errors.append("❌ At least one search URL is required")
       
        if errors:
            for error in errors:
                st.error(error)
            log_to_debug(f"❌ LinkedIn scraping validation failed: {len(errors)} errors")
        else:
            # All validations passed - start scraping
            log_to_debug(f"🚀 Starting LinkedIn scraping: {len(scrape_urls)} URLs, {posts_per_url} posts each")
           
            with st.spinner(f"🔄 Scraping LinkedIn... This may take several minutes (estimated {len(scrape_urls) * posts_per_url} posts)"):
                try:
                    # Import the scraper
                    from ekjdcnskldnc import LinkedInLeadScraper
                   
                    # Create progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                   
                    status_text.text("🔐 Logging into LinkedIn...")
                    progress_bar.progress(10)
                   
                    # Initialize scraper
                    scraper = LinkedInLeadScraper(
                        email=linkedin_config["linkedin_email"],
                        password=linkedin_config["linkedin_password"],
                        apollo_api_key=linkedin_config["apollo_api_key"] if linkedin_config["enrich_emails"] else None,
                        mistral_api_key=main_llm_config["api_key"]  # Use API key from Tab 1
                    )
                   
                    log_to_debug("✅ LinkedIn scraper initialized")
                   
                    # Login
                    scraper.login()
                    log_to_debug("✅ LinkedIn login successful")
                   
                    status_text.text("🔍 Scraping posts from URLs...")
                    progress_bar.progress(30)
                   
                    # Scrape URLs
                    all_leads = scraper.scrape_multiple_urls(
                        urls=scrape_urls,
                        posts_per_url=posts_per_url
                    )
                   
                    log_to_debug(f"✅ Scraped {len(all_leads)} posts")
                   
                    status_text.text("💾 Processing and enriching data...")
                    progress_bar.progress(70)
                   
                    # Save to CSV with enrichment options
                    csv_filename = scraper.save_to_csv(
                        data=all_leads,
                        enrich_with_email=linkedin_config["enrich_emails"],
                        filter_with_mistral=True,  # Always filter posts
                        analyze_companies=linkedin_config["enrich_industries"]
                    )
                   
                    # Close browser
                    scraper.close()
                    log_to_debug("✅ Browser closed")
                   
                    progress_bar.progress(100)
                    status_text.text("✅ Scraping complete!")
                   
                    # Load the CSV and store in session state
                    if csv_filename and os.path.exists(csv_filename):
                        try:
                            scraped_df = pd.read_csv(csv_filename)
                            st.session_state.linkedin_scraped_data = {
                                'df': scraped_df,
                                'filename': csv_filename
                            }
                            log_to_debug(f"✅ Loaded CSV: {csv_filename} ({len(scraped_df)} rows)")
                            st.success(f"✅ Successfully scraped {len(scraped_df)} posts!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Error loading CSV: {str(e)}")
                            log_to_debug(f"❌ Error loading CSV: {str(e)}")
                    else:
                        st.warning("⚠️ Scraping completed but no data file was generated")
                        log_to_debug("⚠️ No CSV file generated")
               
                except ImportError:
                    st.error("❌ LinkedIn scraper module not found. Please ensure 'ekjdcnskldnc.py' is in the same directory.")
                    log_to_debug("❌ Import error: ekjdcnskldnc.py not found")
               
                except Exception as e:
                    st.error(f"❌ Error during scraping: {str(e)}")
                    log_to_debug(f"❌ LinkedIn scraping error: {str(e)}")
                    import traceback
                    with st.expander("🔍 Error Details"):
                        st.code(traceback.format_exc())
                   
                    # Try to close browser if it's open
                    try:
                        if 'scraper' in locals():
                            scraper.close()
                    except:
                        pass
   
    # Display scraped data if available
    if st.session_state.linkedin_scraped_data:
        st.markdown("---")
        st.markdown("### 📊 Scraped Data Results")
       
        scraped_df = st.session_state.linkedin_scraped_data['df']
        csv_filename = st.session_state.linkedin_scraped_data['filename']
       
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
       
        with col1:
            st.metric("Total Posts", len(scraped_df))
       
        with col2:
            # Count emails
            email_col = None
            for col in scraped_df.columns:
                if 'email' in col.lower() and 'status' not in col.lower():
                    email_col = col
                    break
            email_count = scraped_df[email_col].notna().sum() if email_col else 0
            st.metric("With Email", email_count)
       
        with col3:
            # Count verified emails
            if 'email_verification_status' in scraped_df.columns:
                verified_count = (scraped_df['email_verification_status'] == 'verified').sum()
                st.metric("Verified Emails", verified_count)
            else:
                st.metric("Verified Emails", "N/A")
       
        with col4:
            # Count industry enriched
            if 'company_industry' in scraped_df.columns:
                industry_count = scraped_df['company_industry'].notna().sum()
                st.metric("With Industry", industry_count)
            else:
                st.metric("With Industry", "N/A")
       
        # Display dataframe
        st.markdown("#### 📋 Data Preview")
        st.dataframe(
            scraped_df,
            width='stretch',
            height=400
        )
       
        # Download button
        csv_data = scraped_df.to_csv(index=False).encode('utf-8')
       
        st.download_button(
            label="⬇️ Download Scraped Data (CSV)",
            data=csv_data,
            file_name=csv_filename,
            mime="text/csv",
            width='stretch',
            key="download_linkedin_csv"
        )
       
        # Optional: Upload to Google Sheets
        if st.session_state.get("gsheet_authenticated", False) and st.session_state.get("gsheet_service"):
            st.markdown("---")
            st.markdown("### ☁️ Upload to Google Sheets")
           
            sheet_name = f"LinkedIn Scrape {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            custom_sheet_name = st.text_input(
                "Customize Sheet Name (optional):",
                value=sheet_name,
                key="linkedin_gsheet_name"
            )
           
            if st.button("📤 Upload to Google Sheets", type="primary", width='stretch', key="upload_linkedin_gsheet"):
                log_to_debug(f"Starting upload to Google Sheets: {custom_sheet_name}")
               
                try:
                    with st.spinner("Uploading to Google Sheets..."):
                        service = st.session_state.gsheet_service
                        spreadsheet_url = upload_to_google_sheets(service, scraped_df, custom_sheet_name)
                       
                        log_to_debug(f"✅ Successfully uploaded to Google Sheets: {spreadsheet_url}")
                        st.success(f"✅ Successfully uploaded to Google Sheets!")
                        st.markdown(f"🔗 **[Open Spreadsheet]({spreadsheet_url})**")
               
                except Exception as e:
                    log_to_debug(f"❌ Upload to Google Sheets failed: {str(e)}")
                    st.error(f"❌ Upload failed: {str(e)}")
                    import traceback
                    with st.expander("🔍 Error Details"):
                        st.code(traceback.format_exc())
   
    # Render Mail Sender Section for Tab 2
    render_mail_sender_section("(Tab 2)")
with tab3:
    st.title("🔄 CSV Deduplicator & LMS Formatter")
    st.markdown("Upload two CSV files to remove duplicates and format for LMS")
    
    # Create two sub-sections
    section = st.radio(
        "Choose Operation:",
        ["🔄 Deduplicator", "📋 LMS Formatter"],
        horizontal=True,
        key="tab3_section_selector"
    )
    
    st.markdown("---")
    
    # ============ SECTION 1: DEDUPLICATOR ============
    if section == "🔄 Deduplicator":
        st.markdown("### 📤 Upload CSV Files")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📄 CSV File 1")
            csv1 = st.file_uploader(
                "Upload first CSV",
                type=["csv"],
                key="dedup_csv1"
            )
            
            if csv1:
                try:
                    df1 = pd.read_csv(csv1)
                    st.success(f"✅ Loaded: {len(df1)} rows")
                    with st.expander("📋 Preview CSV 1", expanded=False):
                        st.dataframe(df1.head(10), width='stretch')
                except Exception as e:
                    st.error(f"❌ Error reading CSV 1: {str(e)}")
                    df1 = None
        
        with col2:
            st.markdown("#### 📄 CSV File 2")
            csv2 = st.file_uploader(
                "Upload second CSV",
                type=["csv"],
                key="dedup_csv2"
            )
            
            if csv2:
                try:
                    df2 = pd.read_csv(csv2)
                    st.success(f"✅ Loaded: {len(df2)} rows")
                    with st.expander("📋 Preview CSV 2", expanded=False):
                        st.dataframe(df2.head(10), width='stretch')
                except Exception as e:
                    st.error(f"❌ Error reading CSV 2: {str(e)}")
                    df2 = None
        
        st.markdown("---")
        
        # Deduplication settings
        st.markdown("### ⚙️ Deduplication Settings")
        
        if csv1 and csv2 and 'df1' in locals() and 'df2' in locals() and df1 is not None and df2 is not None:
            # Get common columns between both CSVs
            common_columns = list(set(df1.columns) & set(df2.columns))
            
            if common_columns:
                dedup_column = st.selectbox(
                    "Select column to match duplicates:",
                    options=common_columns,
                    help="Choose the column to use for identifying duplicate companies",
                    key="dedup_column_select"
                )
                
                case_sensitive = st.checkbox(
                    "Case sensitive matching",
                    value=False,
                    key="case_sensitive_dedup"
                )
                
                st.markdown("---")
                
                if st.button("🔄 Remove Duplicates", type="primary", width='stretch', key="dedup_button"):
                    log_to_debug(f"Starting deduplication on column: {dedup_column}")
                    
                    with st.spinner("Removing duplicates..."):
                        try:
                            # Create copies for deduplication
                            df1_dedup = df1.copy()
                            df2_dedup = df2.copy()
                            
                            # Handle case sensitivity
                            if not case_sensitive:
                                df1_compare = df1_dedup[dedup_column].astype(str).str.lower().str.strip()
                                df2_compare = df2_dedup[dedup_column].astype(str).str.lower().str.strip()
                            else:
                                df1_compare = df1_dedup[dedup_column].astype(str).str.strip()
                                df2_compare = df2_dedup[dedup_column].astype(str).str.strip()
                            
                            # Find duplicates
                            duplicates_mask = df1_compare.isin(df2_compare)
                            duplicates_count = duplicates_mask.sum()
                            
                            # Remove duplicates from CSV 1
                            df1_cleaned = df1_dedup[~duplicates_mask].reset_index(drop=True)
                            
                            # Store in session state
                            st.session_state.dedup_result = df1_cleaned
                            st.session_state.dedup_stats = {
                                'original_count': len(df1),
                                'duplicates_removed': duplicates_count,
                                'final_count': len(df1_cleaned)
                            }
                            
                            log_to_debug(f"✅ Deduplication complete: {duplicates_count} duplicates removed")
                            st.success(f"✅ Deduplication complete! Removed {duplicates_count} duplicates")
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"❌ Error during deduplication: {str(e)}")
                            log_to_debug(f"❌ Deduplication error: {str(e)}")
            else:
                st.warning("⚠️ No common columns found between the two CSV files")
        
        else:
            st.info("💡 Please upload both CSV files to proceed with deduplication")
        
        # Display results
        if 'dedup_result' in st.session_state and st.session_state.dedup_result is not None:
            st.markdown("---")
            st.markdown("### 📊 Deduplication Results")

            stats = st.session_state.dedup_stats

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Original Records", stats['original_count'])

            with col2:
                st.metric("Duplicates Removed", stats['duplicates_removed'])

            with col3:
                st.metric("Final Records", stats['final_count'])

            st.markdown("#### 📋 Cleaned Data Preview")
            st.dataframe(st.session_state.dedup_result, width='stretch')

            cleaned_csv = st.session_state.dedup_result.to_csv(index=False)
            st.download_button(
                label="⬇️ Download Deduplicated CSV",
                data=cleaned_csv,
                file_name=f"deduplicated_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                width='stretch',
                key="download_dedup_csv"
            )
    
    # ============ SECTION 2: LMS FORMATTER ============
    else:  # LMS Formatter section
        st.markdown("### 📋 LMS Format Converter")
        st.markdown("Convert your CSV to LMS-compatible format with smart column mapping")
        
        st.info("""
        **Target LMS Format:**
        `Industry` | `Company ` | `Name ` | `LinkedIn profile URL` | 
        `Designation` | `Email` | `Phone No.` | `Post Url`
        """)
        
        st.markdown("---")
        
        # File upload
        st.markdown("#### 📤 Upload CSV File")
        lms_csv = st.file_uploader(
            "Upload CSV to format for LMS",
            type=["csv"],
            key="lms_formatter_csv"
        )
        
        # Persist uploaded file across reruns
        if lms_csv:
            if 'lms_uploaded_df' not in st.session_state or st.session_state.get('lms_file_name') != lms_csv.name:
                try:
                    st.session_state.lms_uploaded_df = pd.read_csv(lms_csv)
                    st.session_state.lms_file_name = lms_csv.name
                    log_to_debug(f"✅ LMS CSV loaded: {lms_csv.name} - {len(st.session_state.lms_uploaded_df)} rows")
                except Exception as e:
                    st.error(f"❌ Error reading CSV: {str(e)}")
                    log_to_debug(f"❌ LMS CSV read error: {str(e)}")
                    st.session_state.lms_uploaded_df = None
        
        # Work with persisted dataframe
        if 'lms_uploaded_df' in st.session_state and st.session_state.lms_uploaded_df is not None:
            df_lms = st.session_state.lms_uploaded_df
            
            st.success(f"✅ Loaded: {len(df_lms)} rows, {len(df_lms.columns)} columns")
            
            with st.expander("📋 Preview Uploaded Data", expanded=False):
                st.dataframe(df_lms.head(10), width='stretch')
                st.write("**Columns detected:**", list(df_lms.columns))
            
            st.markdown("---")
            
            # Smart column mapping with AI
            st.markdown("### 🤖 Smart Column Mapping")
            st.caption("AI will automatically map your columns to LMS format")
            
            if st.button("🚀 Auto-Map Columns with AI", type="primary", width='stretch', key="auto_map_btn"):
                # Check if LLM is configured
                main_llm_config = st.session_state.llm_config
                
                if not main_llm_config.get("validated") or not main_llm_config.get("api_key"):
                    st.error("❌ Please configure and validate your LLM API key in Tab 1 first")
                    log_to_debug("❌ LMS Formatter: No validated LLM configuration")
                else:
                    with st.spinner("🤖 AI is analyzing your columns and mapping them..."):
                        try:
                            # Prepare column mapping request
                            source_columns = list(df_lms.columns)
                            target_columns = [
                                "Industry/Sector",
                                "Company Name",
                                "Author/Founder Name",
                                "LinkedIn profile URL",
                                "Designation",
                                "Email",
                                "Phone No.",
                                "Post Url"
                            ]
                            
                            # Create AI prompt
                            prompt = f"""You are a data mapping expert. Map the source CSV columns to the target LMS format columns.

SOURCE COLUMNS:
{', '.join(source_columns)}

TARGET LMS FORMAT COLUMNS:
{', '.join(target_columns)}

INSTRUCTIONS:
1. Map each TARGET column to the most appropriate SOURCE column
2. If no good match exists, return "EMPTY" for that target column
3. Be smart about variations (e.g., "Founder_Name" maps to "Author/Founder Name")
4. Consider common variations like:
   - Company/Company_Name/CompanyName → Company Name
   - Founder/Author/Founder_Name → Author/Founder Name
   - LinkedIn/LinkedIn_URL/Profile → LinkedIn profile URL
   - Industry/Sector/Industry_Sector → Industry/Sector
   - Email/Email_Address → Email
   - Phone/Phone_Number/Contact → Phone No.
   - Post/Post_Link/URL → Post Url

Return ONLY a JSON object with this exact structure:
{{
  "Industry/Sector": "source_column_name or EMPTY",
  "Company Name": "source_column_name or EMPTY",
  "Author/Founder Name": "source_column_name or EMPTY",
  "LinkedIn profile URL": "source_column_name or EMPTY",
  "Designation": "source_column_name or EMPTY",
  "Email": "source_column_name or EMPTY",
  "Phone No.": "source_column_name or EMPTY",
  "Post Url": "source_column_name or EMPTY"
}}

Return ONLY the JSON, no explanations."""

                            log_to_debug(f"🤖 Sending column mapping request to {main_llm_config['provider']}")
                            
                            # Call LLM API
                            provider = main_llm_config["provider"]
                            api_key = main_llm_config["api_key"]
                            endpoint = get_active_endpoint(main_llm_config, 1)
                            
                            if provider == "Mistral":
                                response = requests.post(
                                    endpoint,
                                    headers={"Authorization": f"Bearer {api_key}"},
                                    json={
                                        "model": "mistral-small-latest",
                                        "messages": [{"role": "user", "content": prompt}],
                                        "temperature": 0.1
                                    },
                                    timeout=30
                                )
                                
                                if response.status_code == 200:
                                    result = response.json()
                                    mapping_text = result['choices'][0]['message']['content'].strip()
                                    
                                    # Extract JSON from response
                                    import re
                                    json_match = re.search(r'\{.*\}', mapping_text, re.DOTALL)
                                    if json_match:
                                        mapping_json = json.loads(json_match.group())
                                        st.session_state.lms_column_mapping = mapping_json
                                        log_to_debug(f"✅ AI column mapping successful: {mapping_json}")
                                        st.success("✅ AI successfully mapped your columns!")
                                        st.rerun()
                                    else:
                                        st.error("❌ Could not parse AI response")
                                        log_to_debug(f"❌ Failed to parse JSON from: {mapping_text}")
                                else:
                                    st.error(f"❌ API Error: {response.status_code}")
                                    log_to_debug(f"❌ API error: {response.text}")
                            
                            elif provider == "OpenAI":
                                response = requests.post(
                                    endpoint,
                                    headers={"Authorization": f"Bearer {api_key}"},
                                    json={
                                        "model": "gpt-3.5-turbo",
                                        "messages": [{"role": "user", "content": prompt}],
                                        "temperature": 0.1
                                    },
                                    timeout=30
                                )
                                
                                if response.status_code == 200:
                                    result = response.json()
                                    mapping_text = result['choices'][0]['message']['content'].strip()
                                    
                                    import re
                                    json_match = re.search(r'\{.*\}', mapping_text, re.DOTALL)
                                    if json_match:
                                        mapping_json = json.loads(json_match.group())
                                        st.session_state.lms_column_mapping = mapping_json
                                        log_to_debug(f"✅ AI column mapping successful: {mapping_json}")
                                        st.success("✅ AI successfully mapped your columns!")
                                        st.rerun()
                                    else:
                                        st.error("❌ Could not parse AI response")
                                        log_to_debug(f"❌ Failed to parse JSON from: {mapping_text}")
                                else:
                                    st.error(f"❌ API Error: {response.status_code}")
                                    log_to_debug(f"❌ API error: {response.text}")
                            
                            else:
                                st.warning(f"⚠️ Auto-mapping not yet supported for {provider}. Please use manual mapping below.")
                        
                        except Exception as e:
                            st.error(f"❌ Error during AI mapping: {str(e)}")
                            log_to_debug(f"❌ LMS AI mapping error: {str(e)}")
                            import traceback
                            with st.expander("🔍 Error Details"):
                                st.code(traceback.format_exc())
            
            # Display mapping results or manual mapping
            if 'lms_column_mapping' in st.session_state:
                st.markdown("---")
                st.markdown("### ✅ Column Mapping Results")
                
                mapping = st.session_state.lms_column_mapping
                
                # Show mapping table
                mapping_df = pd.DataFrame([
                    {"Target Column": target, "Mapped From": source if source != "EMPTY" else "⚠️ Not mapped"}
                    for target, source in mapping.items()
                ])
                
                st.dataframe(mapping_df, width='stretch')
                
                # Manual override option
                with st.expander("✏️ Manual Override (Optional)", expanded=False):
                    st.caption("Adjust mappings if needed")
                    
                    manual_mapping = {}
                    for target_col in mapping.keys():
                        options = ["EMPTY"] + list(df_lms.columns)
                        current_value = mapping.get(target_col, "EMPTY")
                        
                        if current_value not in options:
                            current_value = "EMPTY"
                        
                        selected = st.selectbox(
                            f"{target_col}:",
                            options=options,
                            index=options.index(current_value),
                            key=f"manual_map_{target_col}"
                        )
                        manual_mapping[target_col] = selected
                    
                    if st.button("💾 Save Manual Mappings", width='stretch'):
                        st.session_state.lms_column_mapping = manual_mapping
                        st.success("✅ Manual mappings saved!")
                        st.rerun()
                
                # Generate LMS formatted CSV
                st.markdown("---")
                if st.button("📋 Generate LMS Format CSV", type="primary", width='stretch', key="generate_lms_btn"):
                    with st.spinner("Generating LMS format..."):
                        try:
                            # Create new dataframe with LMS format
                            lms_formatted = pd.DataFrame()
                            
                            for target_col, source_col in mapping.items():
                                if source_col == "EMPTY" or source_col not in df_lms.columns:
                                    # Empty column - no N/A, just empty strings
                                    lms_formatted[target_col] = [""] * len(df_lms)
                                else:
                                    # Map the column and replace NaN with empty string
                                    lms_formatted[target_col] = df_lms[source_col].fillna("")
                            
                            # Store in session state
                            st.session_state.lms_formatted_result = lms_formatted
                            
                            log_to_debug(f"✅ LMS format generated: {len(lms_formatted)} rows")
                            st.success(f"✅ LMS format generated successfully!")
                            st.rerun()
                        
                        except Exception as e:
                            st.error(f"❌ Error generating LMS format: {str(e)}")
                            log_to_debug(f"❌ LMS generation error: {str(e)}")
                            import traceback
                            with st.expander("🔍 Error Details"):
                                st.code(traceback.format_exc())
            
            # Display formatted result - MOVED INSIDE THE MAIN IF BLOCK
            if 'lms_formatted_result' in st.session_state and st.session_state.lms_formatted_result is not None:
                st.markdown("---")
                st.markdown("### 📊 LMS Formatted Data")
                
                lms_result = st.session_state.lms_formatted_result
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Records", len(lms_result))
                with col2:
                    filled_cols = sum([1 for col in lms_result.columns if lms_result[col].astype(str).str.strip().ne("").any()])
                    st.metric("Filled Columns", f"{filled_cols}/{len(lms_result.columns)}")
                with col3:
                    empty_cols = len(lms_result.columns) - filled_cols
                    st.metric("Empty Columns", empty_cols)
                
                st.markdown("#### 📋 Preview")
                st.dataframe(lms_result.head(20), width='stretch')
                
                # Download button
                lms_csv_data = lms_result.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️ Download LMS Formatted CSV",
                    data=lms_csv_data,
                    file_name=f"lms_formatted_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    width='stretch',
                    key="download_lms_csv"
                )
                
                # Clear button
                if st.button("🗑️ Clear and Start Over", width='stretch'):
                    if 'lms_column_mapping' in st.session_state:
                        del st.session_state.lms_column_mapping
                    if 'lms_formatted_result' in st.session_state:
                        del st.session_state.lms_formatted_result
                    if 'lms_uploaded_df' in st.session_state:
                        del st.session_state.lms_uploaded_df
                    if 'lms_file_name' in st.session_state:
                        del st.session_state.lms_file_name
                    st.rerun()
        
        else:
            st.info("💡 Please upload a CSV file to begin LMS formatting")
        # ============ GOOGLE SHEETS INTEGRATION SECTION ============
    st.markdown("---")
    st.markdown("### ☁️ Google Sheets Upload")
    st.caption("Upload your CSV data directly to Google Sheets")
    
    # Initialize session state for sheets config
    if "sheets_config" not in st.session_state:
        st.session_state.sheets_config = {
            "oauth_credentials": None,
            "sheet_url": "https://docs.google.com/spreadsheets/d/1xzfD2jZCHczCsJ6yKdLwazfUXXMyTi9hk3sZa8ps_gE/edit?gid=0#gid=0",
            "sheet_name": "Sheet1",
            "use_lms_csv": True,
            "custom_csv": None
        }
    
    sheets_config = st.session_state.sheets_config
    
    # OAuth Credentials Upload
    st.markdown("#### 🔐 Google OAuth Credentials")
    oauth_file = st.file_uploader(
        "Upload Google OAuth JSON file",
        type=["json"],
        key="sheets_oauth_uploader",
        help="Upload your Google OAuth 2.0 Client ID JSON file"
    )
    
    if oauth_file:
        try:
            oauth_data = json.load(oauth_file)
            if "installed" in oauth_data or "web" in oauth_data:
                sheets_config["oauth_credentials"] = oauth_data
                
                # Save to temporary file for sheets.py
                with open('temp_sheets_credentials.json', 'w') as f:
                    json.dump(oauth_data, f)
                
                st.success("✅ OAuth credentials loaded successfully")
                log_to_debug("Google Sheets OAuth credentials loaded")
            else:
                st.error("❌ Invalid OAuth JSON structure")
        except Exception as e:
            st.error(f"❌ Error reading OAuth file: {str(e)}")
    
    st.markdown("---")
    
    # CSV Selection
    st.markdown("#### 📄 Select CSV to Upload")
    
    csv_source = st.radio(
        "Choose CSV source:",
        options=["📋 Use LMS Formatted CSV", "📁 Upload Custom CSV"],
        horizontal=True,
        key="sheets_csv_source"
    )
    
    csv_to_upload = None
    csv_filename = None
    
    if csv_source == "📋 Use LMS Formatted CSV":
        if 'lms_formatted_result' in st.session_state and st.session_state.lms_formatted_result is not None:
            csv_to_upload = st.session_state.lms_formatted_result
            csv_filename = "lms_formatted_temp.csv"
            
            # Save to temporary file
            csv_to_upload.to_csv(csv_filename, index=False)
            
            st.success(f"✅ Using LMS formatted CSV ({len(csv_to_upload)} rows)")
            
            with st.expander("📋 Preview Data", expanded=False):
                st.dataframe(csv_to_upload.head(10), width='stretch')
        else:
            st.warning("⚠️ No LMS formatted CSV available. Please format a CSV first.")
    
    else:  # Upload Custom CSV
        custom_csv = st.file_uploader(
            "Upload CSV file",
            type=["csv"],
            key="sheets_custom_csv_uploader"
        )
        
        if custom_csv:
            try:
                csv_to_upload = pd.read_csv(custom_csv)
                csv_filename = "custom_upload_temp.csv"
                
                # Save to temporary file
                csv_to_upload.to_csv(csv_filename, index=False)
                
                st.success(f"✅ Custom CSV loaded ({len(csv_to_upload)} rows)")
                
                with st.expander("📋 Preview Data", expanded=False):
                    st.dataframe(csv_to_upload.head(10), width='stretch')
            
            except Exception as e:
                st.error(f"❌ Error reading CSV: {str(e)}")
                csv_to_upload = None
    
    st.markdown("---")
    
    # Google Sheets Configuration
    st.markdown("#### 🔗 Google Sheets Configuration")
    
    # Sheet URL with edit option
    sheet_url_editable = st.checkbox(
        "✏️ Edit Sheet URL",
        value=False,
        key="edit_sheet_url_checkbox"
    )
    
    if sheet_url_editable:
        sheet_url = st.text_input(
            "Google Sheets URL",
            value=sheets_config["sheet_url"],
            key="sheets_url_input",
            help="Enter the full Google Sheets URL"
        )
        sheets_config["sheet_url"] = sheet_url
        
        # If URL is edited, ask for sheet name
        sheet_name = st.text_input(
            "Sheet Name (Tab Name)",
            value=sheets_config["sheet_name"],
            key="sheets_name_input",
            help="Enter the name of the tab/sheet within the spreadsheet"
        )
        sheets_config["sheet_name"] = sheet_name
    else:
        st.info(f"📊 Using default sheet: `{sheets_config['sheet_url'][:50]}...`")
        st.caption(f"Sheet Name: `{sheets_config['sheet_name']}`")
    
    st.markdown("---")
    
    # Upload Button
    col_upload, col_status = st.columns([1, 2])
    
    with col_upload:
        upload_to_sheets = st.button(
            "☁️ Upload to Google Sheets",
            type="primary",
            width='stretch',
            key="upload_to_sheets_btn"
        )
    
    # Validation and Upload
    if upload_to_sheets:
        errors = []
        
        if not sheets_config.get("oauth_credentials"):
            errors.append("❌ Please upload Google OAuth credentials")
        
        if csv_to_upload is None:
            errors.append("❌ Please select or upload a CSV file")
        
        if not sheets_config.get("sheet_url"):
            errors.append("❌ Please provide a Google Sheets URL")
        
        if not sheets_config.get("sheet_name"):
            errors.append("❌ Please provide a Sheet Name")
        
        if errors:
            for error in errors:
                st.error(error)
        else:
            with st.spinner("☁️ Uploading to Google Sheets... This may take a moment"):
                try:
                    # Import the sheets upload function
                    from sheet import authenticate, upload_csv_to_sheet, extract_spreadsheet_id
                    
                    log_to_debug(f"🚀 Starting Google Sheets upload")
                    log_to_debug(f"   CSV: {csv_filename} ({len(csv_to_upload)} rows)")
                    log_to_debug(f"   Sheet URL: {sheets_config['sheet_url']}")
                    log_to_debug(f"   Sheet Name: {sheets_config['sheet_name']}")
                    
                    # Extract spreadsheet ID
                    spreadsheet_id = extract_spreadsheet_id(sheets_config["sheet_url"])
                    log_to_debug(f"   Spreadsheet ID: {spreadsheet_id}")
                    
                    # Authenticate
                    service = authenticate('temp_sheets_credentials.json')
                    
                    # Upload CSV
                    result_url, cells_updated = upload_csv_to_sheet(
                        service=service,
                        spreadsheet_id=spreadsheet_id,
                        csv_path=csv_filename,
                        sheet_name=sheets_config["sheet_name"]
                    )
                    
                    log_to_debug(f"✅ Upload successful: {cells_updated} cells updated")
                    
                    st.success(f"✅ Successfully uploaded to Google Sheets!")
                    st.success(f"📊 {cells_updated} cells uploaded")
                    st.markdown(f"🔗 **[Open Spreadsheet]({result_url})**")
                    
                    # Clean up temporary files
                    try:
                        if os.path.exists(csv_filename):
                            os.remove(csv_filename)
                        if os.path.exists('temp_sheets_credentials.json'):
                            os.remove('temp_sheets_credentials.json')
                    except:
                        pass
                
                except ImportError:
                    st.error("❌ sheets.py module not found. Please ensure it's in the same directory.")
                    log_to_debug("❌ Import error: sheets.py not found")
                
                except Exception as e:
                    st.error(f"❌ Upload failed: {str(e)}")
                    log_to_debug(f"❌ Google Sheets upload error: {str(e)}")
                    
                    import traceback
                    with st.expander("🔍 Error Details"):
                        st.code(traceback.format_exc())
                    
                    # Clean up temporary files
                    try:
                        if csv_filename and os.path.exists(csv_filename):
                            os.remove(csv_filename)
                        if os.path.exists('temp_sheets_credentials.json'):
                            os.remove('temp_sheets_credentials.json')
                    except:
                        pass
    
        st.markdown("---")
    

with tab4:
    st.title("📊 Lead Management Dashboard")
    st.markdown("Track and analyze your email campaigns")
    
    # Initialize database on first run
    initialize_leads_database()
    update_database_schema()
    
    # ===== AUTO-REFRESH SYSTEM (SILENT BACKGROUND) =====
    import time
    
    AUTO_REFRESH_INTERVAL = 5  # seconds
    
    # Initialize auto-refresh state
    if "last_auto_refresh_time" not in st.session_state:
        st.session_state.last_auto_refresh_time = time.time()
    
    # Check if it's time to auto-refresh (SILENT - runs in background)
    current_time = time.time()
    time_since_refresh = current_time - st.session_state.last_auto_refresh_time
    
    if time_since_refresh >= AUTO_REFRESH_INTERVAL:
        st.session_state.last_auto_refresh_time = current_time
        
        # Check for replies silently in background
        try:
            replies_found = check_email_replies()
            if replies_found > 0:
                log_to_debug(f"🔄 Auto-refresh found {replies_found} new replies")
        except Exception as e:
            log_to_debug(f"⚠️ Auto-refresh error: {str(e)}")
        
        # Trigger silent rerun
        st.rerun()
    
    # Single Refresh button with auto-refresh text below
    col_refresh, col_spacer = st.columns([1, 5])
    
    with col_refresh:
        if st.button("🔄 Refresh Data", width='stretch', key="refresh_dashboard"):
            with st.spinner("🔄 Refreshing data and checking for replies..."):
                # Check for replies first
                replies_found = check_email_replies()
                
                # Show notification based on results
                if replies_found > 0:
                    st.success(f"✅ Found {replies_found} new replies!")
                else:
                    st.toast("📭 No new replies found", icon="ℹ️")
                
                # Wait a moment for user to see the message
                time.sleep(0.5)
                
                # Reset auto-refresh timer so it doesn't trigger immediately
                st.session_state.last_auto_refresh_time = time.time()
                
                # Rerun to refresh the dashboard
                st.rerun()
        
        # Small text showing auto-refresh countdown
        next_refresh = int(AUTO_REFRESH_INTERVAL - time_since_refresh)
        if next_refresh > 0:
            st.caption(f"Auto-refresh in {next_refresh}s")
        else:
            st.caption("Auto-refreshing...")
    
    with st.expander("🔍 View Debug Logs", expanded=False):
        if "debug_logs" in st.session_state and st.session_state.debug_logs:
            # Show last 50 logs
            recent_logs = st.session_state.debug_logs[-50:]
            for log in reversed(recent_logs):
                st.text(log)
        else:
            st.info("No debug logs available yet")
    
    st.markdown("---")
    
    # Load data
    df = load_leads_data()
    
    if df.empty:
        st.info("📭 No lead data available yet. Send some emails to see analytics!")
        st.markdown("### 💡 Getting Started")
        st.markdown("""
        1. Go to **Tab 1** or **Tab 2** to generate leads
        2. Use the **Mail Sender Section** to send emails
        3. Come back here to track your campaign performance
        """)
    else:
        # Metrics
        st.markdown("### 📊 Campaign Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        total_sent = len(df)
        opened_count = df['opened_at'].notnull().sum() if 'opened_at' in df.columns else 0
        replied_count = df['replied'].sum() if 'replied' in df.columns else 0
        bounced_count = df['bounced'].sum() if 'bounced' in df.columns else 0
        
        with col1:
            st.metric("📧 Total Sent", total_sent)
        
        with col2:
            open_rate = (opened_count / total_sent * 100) if total_sent > 0 else 0
            st.metric("👁️ Opened", opened_count, f"{open_rate:.1f}%")
        
        with col3:
            reply_rate = (replied_count / total_sent * 100) if total_sent > 0 else 0
            st.metric("💬 Replied", replied_count, f"{reply_rate:.1f}%")
        
        with col4:
            bounce_rate = (bounced_count / total_sent * 100) if total_sent > 0 else 0
            st.metric("⚠️ Bounced", bounced_count, f"{bounce_rate:.1f}%")
        
        st.markdown("---")
        
        # ============ CHART VISUALIZATIONS ============
        st.markdown("### 📈 Visual Analytics")
        
        # Row 1: Overall Performance Charts
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            # Funnel Chart - Email Journey
            st.markdown("#### 🔽 Email Engagement Funnel")
            funnel_data = pd.DataFrame({
                'Stage': ['Sent', 'Delivered', 'Opened', 'Replied'],
                'Count': [
                    total_sent,
                    total_sent - bounced_count,
                    opened_count,
                    replied_count
                ]
            })
            
            fig_funnel = go.Figure(go.Funnel(
                y = funnel_data['Stage'],
                x = funnel_data['Count'],
                textposition = "inside",
                textinfo = "value+percent initial",
                marker = dict(
                    color = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
                ),
                connector = {"line": {"color": "royalblue"}}
            ))
            
            fig_funnel.update_layout(
                height=400,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            
            st.plotly_chart(fig_funnel, use_container_width=True)
        
        with col_chart2:
            # Pie Chart - Email Status Distribution
            st.markdown("#### 🎯 Email Status Distribution")
            
            not_opened = total_sent - opened_count - bounced_count
            
            status_data = pd.DataFrame({
                'Status': ['Replied', 'Opened (No Reply)', 'Sent (Not Opened)', 'Bounced'],
                'Count': [replied_count, opened_count - replied_count, not_opened, bounced_count]
            })
            
            # Remove zero values
            status_data = status_data[status_data['Count'] > 0]
            
            fig_pie = px.pie(
                status_data, 
                values='Count', 
                names='Status',
                color_discrete_sequence=['#2ecc71', '#3498db', '#95a5a6', '#e74c3c'],
                hole=0.4
            )
            
            fig_pie.update_traces(
                textposition='inside', 
                textinfo='percent+label',
                hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
            )
            
            fig_pie.update_layout(
                height=400,
                margin=dict(l=20, r=20, t=40, b=20),
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.2,
                    xanchor="center",
                    x=0.5
                )
            )
            
            st.plotly_chart(fig_pie, use_container_width=True)
        
        st.markdown("---")
        
        # Row 2: Template and Sender Performance
        col_chart3, col_chart4 = st.columns(2)
        
        with col_chart3:
            st.markdown("#### 📧 Performance by Template")
            if 'template_used' in df.columns:
                template_stats = df.groupby('template_used').agg({
                    'email': 'count',
                    'replied': 'sum',
                    'bounced': 'sum'
                }).rename(columns={'email': 'sent'})
                
                template_stats['reply_rate'] = (template_stats['replied'] / template_stats['sent'] * 100).round(1)
                template_stats = template_stats.reset_index()
                
                # Grouped Bar Chart
                fig_template = go.Figure()
                
                fig_template.add_trace(go.Bar(
                    name='Sent',
                    x=template_stats['template_used'],
                    y=template_stats['sent'],
                    marker_color='#3498db',
                    text=template_stats['sent'],
                    textposition='auto'
                ))
                
                fig_template.add_trace(go.Bar(
                    name='Replied',
                    x=template_stats['template_used'],
                    y=template_stats['replied'],
                    marker_color='#2ecc71',
                    text=template_stats['replied'],
                    textposition='auto'
                ))
                
                fig_template.add_trace(go.Bar(
                    name='Bounced',
                    x=template_stats['template_used'],
                    y=template_stats['bounced'],
                    marker_color='#e74c3c',
                    text=template_stats['bounced'],
                    textposition='auto'
                ))
                
                fig_template.update_layout(
                    barmode='group',
                    height=400,
                    xaxis_title="Template",
                    yaxis_title="Count",
                    margin=dict(l=20, r=20, t=40, b=20),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                st.plotly_chart(fig_template, use_container_width=True)
                
                # Reply Rate Line Chart
                st.markdown("##### Reply Rate by Template")
                fig_reply_rate = go.Figure()
                
                fig_reply_rate.add_trace(go.Bar(
                    x=template_stats['template_used'],
                    y=template_stats['reply_rate'],
                    marker_color='#9b59b6',
                    text=template_stats['reply_rate'].apply(lambda x: f'{x:.1f}%'),
                    textposition='auto'
                ))
                
                fig_reply_rate.update_layout(
                    height=300,
                    xaxis_title="Template",
                    yaxis_title="Reply Rate (%)",
                    margin=dict(l=20, r=20, t=20, b=20)
                )
                
                st.plotly_chart(fig_reply_rate, use_container_width=True)
            else:
                st.info("No template data available")
        
        with col_chart4:
            st.markdown("#### 👤 Performance by Sender")
            if 'sender_email' in df.columns:
                sender_stats = df.groupby('sender_email').agg({
                    'email': 'count',
                    'replied': 'sum',
                    'bounced': 'sum'
                }).rename(columns={'email': 'sent'})
                
                sender_stats['reply_rate'] = (sender_stats['replied'] / sender_stats['sent'] * 100).round(1)
                sender_stats = sender_stats.reset_index()
                
                # Truncate long email addresses for display
                sender_stats['sender_short'] = sender_stats['sender_email'].apply(
                    lambda x: x.split('@')[0][:15] + '...' if len(x) > 20 else x
                )
                
                # Grouped Bar Chart
                fig_sender = go.Figure()
                
                fig_sender.add_trace(go.Bar(
                    name='Sent',
                    x=sender_stats['sender_short'],
                    y=sender_stats['sent'],
                    marker_color='#3498db',
                    text=sender_stats['sent'],
                    textposition='auto',
                    hovertext=sender_stats['sender_email']
                ))
                
                fig_sender.add_trace(go.Bar(
                    name='Replied',
                    x=sender_stats['sender_short'],
                    y=sender_stats['replied'],
                    marker_color='#2ecc71',
                    text=sender_stats['replied'],
                    textposition='auto',
                    hovertext=sender_stats['sender_email']
                ))
                
                fig_sender.add_trace(go.Bar(
                    name='Bounced',
                    x=sender_stats['sender_short'],
                    y=sender_stats['bounced'],
                    marker_color='#e74c3c',
                    text=sender_stats['bounced'],
                    textposition='auto',
                    hovertext=sender_stats['sender_email']
                ))
                
                fig_sender.update_layout(
                    barmode='group',
                    height=400,
                    xaxis_title="Sender",
                    yaxis_title="Count",
                    margin=dict(l=20, r=20, t=40, b=20),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                st.plotly_chart(fig_sender, use_container_width=True)
                
                # Reply Rate by Sender
                st.markdown("##### Reply Rate by Sender")
                fig_sender_reply = go.Figure()
                
                fig_sender_reply.add_trace(go.Bar(
                    x=sender_stats['sender_short'],
                    y=sender_stats['reply_rate'],
                    marker_color='#e67e22',
                    text=sender_stats['reply_rate'].apply(lambda x: f'{x:.1f}%'),
                    textposition='auto',
                    hovertext=sender_stats['sender_email']
                ))
                
                fig_sender_reply.update_layout(
                    height=300,
                    xaxis_title="Sender",
                    yaxis_title="Reply Rate (%)",
                    margin=dict(l=20, r=20, t=20, b=20)
                )
                
                st.plotly_chart(fig_sender_reply, use_container_width=True)
            else:
                st.info("No sender data available")
        
        st.markdown("---")
        
        # Row 3: Timeline Analysis
        if 'sent_at' in df.columns:
            st.markdown("### 📅 Timeline Analysis")
            
            # Convert sent_at to datetime
            df['sent_at'] = pd.to_datetime(df['sent_at'])
            df['sent_date'] = df['sent_at'].dt.date
            
            # Daily activity
            daily_stats = df.groupby('sent_date').agg({
                'email': 'count',
                'replied': 'sum',
                'bounced': 'sum'
            }).rename(columns={'email': 'sent'}).reset_index()
            
            fig_timeline = go.Figure()
            
            fig_timeline.add_trace(go.Scatter(
                x=daily_stats['sent_date'],
                y=daily_stats['sent'],
                mode='lines+markers',
                name='Sent',
                line=dict(color='#3498db', width=2),
                marker=dict(size=8)
            ))
            
            fig_timeline.add_trace(go.Scatter(
                x=daily_stats['sent_date'],
                y=daily_stats['replied'],
                mode='lines+markers',
                name='Replied',
                line=dict(color='#2ecc71', width=2),
                marker=dict(size=8)
            ))
            
            fig_timeline.add_trace(go.Scatter(
                x=daily_stats['sent_date'],
                y=daily_stats['bounced'],
                mode='lines+markers',
                name='Bounced',
                line=dict(color='#e74c3c', width=2),
                marker=dict(size=8)
            ))
            
            fig_timeline.update_layout(
                title="Daily Email Activity",
                xaxis_title="Date",
                yaxis_title="Count",
                height=400,
                hovermode='x unified',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig_timeline, use_container_width=True)
        
        st.markdown("---")
        
        # Detailed Data Table
        st.markdown("### 📋 Detailed Lead Data")
        
        # Filter options
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        
        with filter_col1:
            if 'template_used' in df.columns:
                templates = ['All'] + list(df['template_used'].unique())
                selected_template = st.selectbox("Filter by Template", templates)
                if selected_template != 'All':
                    df = df[df['template_used'] == selected_template]
        
        with filter_col2:
            if 'sender_email' in df.columns:
                senders = ['All'] + list(df['sender_email'].unique())
                selected_sender = st.selectbox("Filter by Sender", senders)
                if selected_sender != 'All':
                    df = df[df['sender_email'] == selected_sender]
        
        with filter_col3:
            status_filter = st.selectbox("Filter by Status", ['All', 'Replied', 'Opened', 'Bounced', 'Sent Only'])
            if status_filter == 'Replied':
                df = df[df['replied'] == 1]
            elif status_filter == 'Opened':
                df = df[df['opened_at'].notnull()]
            elif status_filter == 'Bounced':
                df = df[df['bounced'] == 1]
            elif status_filter == 'Sent Only':
                df = df[(df['replied'] == 0) & (df['opened_at'].isnull()) & (df['bounced'] == 0)]
        
        # Display filtered data
        st.dataframe(
            df,
            width='stretch',
            height=400
        )
        st.markdown("---")
        st.markdown("### 💾 Export Data")
        
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download as CSV",
            data=csv_data,
            file_name=f"lead_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
        # Clear data option
        st.markdown("---")
        if st.button("🗑️ Clear All Lead Data", type="secondary", width='stretch'):
            if st.checkbox("⚠️ I understand this will delete all lead tracking data"):
                try:
                    conn = sqlite3.connect("leads.db")
                    cursor = conn.cursor()
                    cursor.execute("DELETE FROM leads")
                    conn.commit()
                    conn.close()
                    st.success("✅ All lead data cleared!")
                    log_to_debug("Lead database cleared")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error clearing data: {str(e)}")
    
    # ============ DATABASE MANAGEMENT SECTION ============
    st.markdown("---")
    st.markdown("### 🗄️ Database Management")
    
    # Database Info Section (First)
    st.markdown("#### 📊 Database Info")
    
    try:
        conn = sqlite3.connect("leads.db")
        cursor = conn.cursor()
        
        # Get database statistics
        cursor.execute("SELECT COUNT(*) FROM leads")
        total_records = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM leads WHERE opened_at IS NOT NULL")
        opened_records = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM leads WHERE replied = 1")
        replied_records = cursor.fetchone()[0]
        
        cursor.execute("SELECT MIN(sent_at), MAX(sent_at) FROM leads")
        date_range = cursor.fetchone()
        
        # Get database file size
        import os
        db_size = os.path.getsize("leads.db") / 1024  # Convert to KB
        
        conn.close()
        
        # Display info in horizontal format
        info_col1, info_col2, info_col3 = st.columns(3)
        
        with info_col1:
            st.metric("Total Records", total_records)
        
        with info_col2:
            st.metric("Database Size", f"{db_size:.1f} KB")
        
        with info_col3:
            if date_range[0] and date_range[1]:
                st.metric("Date Range", f"{date_range[0][:10]} to {date_range[1][:10]}")
            else:
                st.metric("Date Range", "No records")
        
    except Exception as e:
        st.error(f"❌ Cannot access database: {str(e)}")
        log_to_debug(f"❌ Database info error: {str(e)}")
    
    st.markdown("---")
    
    # Reset Database Section (Second)
    st.markdown("#### 🔄 Reset Database")
    st.caption("Clear all lead tracking data and start fresh")
    
    # Safety confirmation checkbox
    confirm_reset = st.checkbox(
        "⚠️ I understand this will permanently delete all lead data",
        key="confirm_reset_checkbox"
    )
    
    # Reset button (disabled unless confirmed)
    if st.button(
        "🗑️ Reset Database",
        type="secondary",
        use_container_width=True,
        disabled=not confirm_reset,
        key="reset_database_button"
    ):
        try:
            conn = sqlite3.connect("leads.db")
            cursor = conn.cursor()
            
            # Get count before deletion
            cursor.execute("SELECT COUNT(*) FROM leads")
            count_before = cursor.fetchone()[0]
            
            # Delete all records
            cursor.execute("DELETE FROM leads")
            
            # Reset auto-increment counter
            cursor.execute("DELETE FROM sqlite_sequence WHERE name='leads'")
            
            conn.commit()
            conn.close()
            
            log_to_debug(f"🗑️ Database reset: {count_before} records deleted")
            st.success(f"✅ Database reset successfully! Deleted {count_before} records")
            
            # Wait a moment then refresh
            time.sleep(1)
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error resetting database: {str(e)}")
            log_to_debug(f"❌ Database reset error: {str(e)}")
            import traceback
            with st.expander("🔍 Error Details"):
                st.code(traceback.format_exc())
        st.markdown("#### 📊 Database Info")
        
        try:
            conn = sqlite3.connect("leads.db")
            cursor = conn.cursor()
            
            # Get database statistics
            cursor.execute("SELECT COUNT(*) FROM leads")
            total_records = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM leads WHERE opened_at IS NOT NULL")
            opened_records = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM leads WHERE replied = 1")
            replied_records = cursor.fetchone()[0]
            
            cursor.execute("SELECT MIN(sent_at), MAX(sent_at) FROM leads")
            date_range = cursor.fetchone()
            
            # Get database file size
            import os
            db_size = os.path.getsize("leads.db") / 1024  # Convert to KB
            
            conn.close()
            
            # Display info in horizontal format
            info_col1, info_col2, info_col3 = st.columns(3)
            
            with info_col1:
                st.metric("Total Records", total_records)
            
            with info_col2:
                st.metric("Database Size", f"{db_size:.1f} KB")
            
            with info_col3:
                if date_range[0] and date_range[1]:
                    st.metric("Date Range", f"{date_range[0][:10]} to {date_range[1][:10]}")
                else:
                    st.metric("Date Range", "No records")
            
        except Exception as e:
            st.error(f"❌ Cannot access database: {str(e)}")
            log_to_debug(f"❌ Database info error: {str(e)}")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <p>🚀 Lead Generation Agent | Powered by AI</p>
        <p>Tushar Yadav</p>
    </div>
    """,
    unsafe_allow_html=True
)