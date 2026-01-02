# 🚀 SERA — Smart Enterprise Research & Outreach Agent

📊 **AI-Powered Lead Generation, Outreach & CRM Analytics Platform**

> **SERA** is an enterprise-grade AI agent that automates **startup discovery, lead enrichment,Personalized LinkedIn lead extraction driven by natural language intent,
 outbound email outreach, and CRM performance tracking** using multi-LLM orchestration, browser automation, and custom-built analytics infrastructure.

---

## 📸 Screenshots

### Dashboard
*Unified dashboard for lead generation, outreach, and analytics*
<img width="1895" height="884" alt="Screenshot 2025-12-30 174754" src="https://github.com/user-attachments/assets/b8d3c3a7-d9a7-49a7-8ddd-7e8de1172a11" />
<img width="1903" height="859" alt="Screenshot 2025-12-30 174811" src="https://github.com/user-attachments/assets/b387136f-5aaa-43f6-b121-3942c443a627" />
<img width="1898" height="817" alt="Screenshot 2025-12-30 174827" src="https://github.com/user-attachments/assets/31d23819-90a5-4653-88b5-5f5fd5be27e5" />


### LLM Configuration
*Multi-LLM provider configuration panel*
<img width="1888" height="884" alt="Screenshot 2025-12-30 175055" src="https://github.com/user-attachments/assets/56caf6b3-fef0-4e6f-a2e7-79356c3bd3b6" />

### Report Preview
*Generated funding report with enriched company data*
<img width="1897" height="882" alt="Screenshot 2025-12-30 180811" src="https://github.com/user-attachments/assets/0d81cfa5-fdde-40e8-a001-e86898908818" />

### Contact Extraction (Apollo Integration)
*Email and phone extraction using Apollo.io API*
<img width="1903" height="888" alt="Screenshot 2025-12-30 180953" src="https://github.com/user-attachments/assets/f527f1f5-519a-46b6-956d-8aef875a0bf9" />
<img width="1890" height="870" alt="Screenshot 2025-12-30 181033" src="https://github.com/user-attachments/assets/3c6c64b0-4783-481a-b4ed-f3464af54562" />
<img width="1889" height="862" alt="Screenshot 2025-12-30 181057" src="https://github.com/user-attachments/assets/2c2f73be-f033-4356-9f7a-c9a16f863049" />

### CRM Analytics Dashboard
*Open rate, reply rate, and bounce rate tracking*
<img width="1893" height="878" alt="Screenshot 2025-12-30 181126" src="https://github.com/user-attachments/assets/f339d03c-df8e-4874-be3c-6642c4e7c043" />

---

## 📖 About SERA

**SERA (Smart Enterprise Research & Outreach Agent)** is a full-stack, production-grade automation platform designed to **discover, enrich, engage, and track leads at scale**.

Unlike traditional lead generation tools, SERA goes beyond data collection by enabling **automated outbound email campaigns** using predefined templates and a **custom-built CRM tracking engine** that monitors:

- 📬 Email open rates  
- 💬 Replies and conversation threads  
- ❌ Bounce rates  
- 📈 Campaign-level engagement metrics  

SERA intelligently analyzes user intent using LLMs to determine whether **weekly or monthly data collection** is required. It orchestrates parallel scraping from curated funding sources, enriches companies via LinkedIn and Apollo.io, launches outreach campaigns, and visualizes engagement metrics — all from a single dashboard.

Email tracking is built **from scratch**, using:
- **1×1 tracking pixels** for open detection  
- **Thread-based reply detection**  
- **Custom bounce handling**  
- **Campaign-level analytics aggregation**

This creates a **closed-loop AI system** that transforms raw web data into **measurable business intelligence**.

---

## ✨ Features

### 🤖 Multi-LLM Intelligence
- Supports **8 LLM providers**:
  - OpenAI
  - Claude
  - Gemini
  - Mistral
  - DeepSeek
  - Qwen
  - Perplexity
  - Llama
- **Dual API key rotation** for reliability and rate-limit handling
- Custom endpoint validation and automatic failover

---

### 📊 Automated Data Collection
- **Weekly Reports** – Scrapes FoundersDay funding announcements  
- **Monthly Reports** – Aggregates data from FoundersDay and GrowthList  
- **Sector Filtering** – 12+ industries (AI, Fintech, Healthcare, EdTech, etc.)  
- **Geographic Coverage** – India & Global markets  

---

### 🔍 Lead Enrichment
- Automated **LinkedIn profile discovery** (founders/CEOs)
- **Contact enrichment** using Apollo.io API
- Industry, category, and company metadata extraction
- Intelligent **duplicate detection and validation**

---

### 🔗 LinkedIn Lead Generation (Personalized)

- Automated **LinkedIn-based lead discovery**
- Accepts **custom search intent** (e.g., role, industry, tech stack, geography)
- Personalized extraction based on user-defined fields:
  - Job title
  - Company size
  - Industry
  - Location
  - Keywords
- AI-assisted profile validation to filter low-quality leads
- Supports CSV upload + direct LinkedIn scraping

---

### 📧 Outreach Automation
- Automated outbound email campaigns using **predefined templates**
- Personalized emails per company
- Thread-based reply tracking
- Custom bounce detection logic

---

### 📊 CRM & Engagement Analytics (Built from Scratch)
- 📬 Open rate tracking using **1×1 pixel technique**
- 💬 Reply detection via **email thread analysis**
- ❌ Bounce rate monitoring
- 📈 Campaign-level performance dashboards
- Real-time analytics visualization

---

### 🎨 User Interface
- Interactive **Streamlit dashboard**
- Real-time progress and debug logs
- Export to **CSV / Excel / Google Sheets**
- OAuth-based Google Sheets integration

---

## 🏗️ Architecture

SERA/
├── app.py # Streamlit UI & analytics dashboard
├── main.py # Core orchestration & LLM routing
├── weekly.py # Weekly funding scraper
├── monthly.py # Monthly funding scraper
├── selenium_extractor.py # LinkedIn & enrichment engine
├── email_engine.py # Outreach automation & templates
├── crm_tracker.py # Open, reply & bounce tracking
├── requirements.txt
└── README.md


---

## 🔄 Data Flow

1. **User Input** → Sector, region, timeframe (Streamlit UI)
2. **Intent Analysis** → LLM determines report frequency
3. **Data Collection**
   - Funding platforms (FoundersDay, GrowthList)
   - LinkedIn personalized lead scraping based on user intent

4. **LLM Structuring**
   - Converts raw web & LinkedIn data into structured lead objects

5. **Enrichment**
   - LinkedIn profile validation
   - Apollo.io contact enrichment

6. **Outreach Automation** → Email campaigns
7. **CRM Tracking** → Open, reply & bounce analytics
8. **Export** → CSV / Excel / Google Sheets

---

## 🛠️ Tech Stack

- **Python**
- **Streamlit**
- **Selenium + BeautifulSoup**
- **Undetected ChromeDriver**
- **Multi-LLM Orchestration**
- **Apollo.io API**
- **Google Sheets API**
- **Pandas**
- **OAuth**
- **Custom CRM Tracking Engine**

---

## 🎯 Use Cases

- 📈 B2B Lead Generation  
- 🧠 Investor & VC Market Research  
- 📊 Startup Intelligence  
- 📧 Automated Outreach Campaigns  
- 🤖 AI Agent System Design Reference  

---

## ⚠️ Limitations

- Rate limits depend on LLM and Apollo.io plans
- LinkedIn scraping may face CAPTCHA challenges
- Optimized for India & Global datasets

---

## 👤 Author

**Tushar Yadav**  
📧 Email: tusharyadav61900@gmail.com  
🔗 LinkedIn: https://www.linkedin.com/in/tushar-yadav-5829bb353/

---

<div align="center">

⭐ **Star SERA if you find it useful!**  
Built with ❤️ for AI-powered automation

</div>
