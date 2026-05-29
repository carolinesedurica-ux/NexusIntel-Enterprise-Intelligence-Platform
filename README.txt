NexusIntel — Enterprise Intelligence Platform
===============================================
Multi-track intelligence powered by Bright Data and Claude AI.
Built for the Bright Data Hackathon — spans all 3 tracks.


THE CORE IDEA
-------------
Give it a company name (e.g. "Salesforce"). It runs three parallel intelligence
pipelines using live web scraping via Bright Data, then Claude AI synthesizes
everything into a unified executive brief.


TRACK 1 — GTM INTELLIGENCE
---------------------------
"What are my competitors doing and who is ready to buy?"

1. Competitor discovery
   SERP searches find companies competing with the target.

2. Competitor profiling
   Scrapes each competitor's pricing/homepage via Web Unlocker (falls back to
   Scraping Browser for JS-heavy sites). Claude extracts pricing tiers,
   messaging themes, and hiring signals.

3. Job posting collection
   Pulls live Indeed job data via Bright Data Web Scraper API to infer
   strategic direction (e.g. heavy ML hiring = AI product push).

4. Buying signal detection
   SERP queries for pain points, vendor-switch intent, and complaints.
   Claude scores each result for purchase intent.

5. Market positioning
   Claude synthesizes everything into a positioning statement and 5 action
   items for a VP of Sales or CMO.


TRACK 2 — FINANCE INTELLIGENCE
--------------------------------
"What is the financial and regulatory risk picture?"

1. Pricing surveillance
   Scrapes the target's /pricing and /plans pages. Claude extracts structured
   plan, price, and tier data.

2. Regulatory monitoring
   SERP queries for fines, enforcement actions, and GDPR/CCPA/SEC news.
   Direct scrape of SEC EDGAR for 8-K filings.

3. Alternative data signals
   Indeed job velocity (headcount growth indicator) + LinkedIn company data
   + news sentiment. Claude interprets these as financial signals
   (e.g. "40 open ML roles = major product investment").

4. Risk scoring
   Claude produces a 0-10 financial risk score and investment summary for a CFO.


TRACK 3 — SECURITY & COMPLIANCE
---------------------------------
"Are we exposed? What do we need to act on?"

1. Threat surface monitoring
   Six SERP query types: data breaches, CVEs, phishing, paste site dumps,
   GitHub code exposure, and malware incidents.

2. Autonomous deep investigation (AI agentic loop)
   When a HIGH or CRITICAL indicator is found, the agent automatically fetches
   that source URL and runs a deeper Claude analysis to surface related
   indicators — no human instruction needed.

3. Compliance monitoring (4 sources)
   Scrapes CISA Advisories, NIST Cybersecurity Framework, GDPR Info, and
   PCI DSS. Every change includes a severity rating and specific action_required.

4. Vendor auto-discovery and risk assessment
   If no vendors are specified, SERP discovers the target's known tech stack.
   Each vendor gets a scored risk profile (0-10) with risk factors and
   a remediation recommendation.

5. Brand and credential exposure scanning
   Six vectors: paste sites, credential dumps, GitHub secrets, typosquatting,
   S3/cloud misconfigs, and raw data file leaks.

6. Webhook alert delivery
   If overall severity is HIGH or CRITICAL, a structured JSON alert is
   automatically POSTed to any configured webhook URL.


CROSS-TRACK SYNTHESIS
----------------------
After all three tracks complete in parallel (LangGraph fan-out), a synthesis
agent combines GTM, Finance, and Security outputs into:

  - A 2-3 sentence C-suite executive summary
  - Top 3 priorities across all tracks
  - Cross-track connections (e.g. competitor hiring in CVE-affected tech stack)
  - Recommended actions


DELIVERY
--------
  Web UI     http://localhost:8001
             Cyberpunk neon dashboard with live API integration

  REST API   POST /intelligence           Run all three tracks
             POST /intelligence/gtm       GTM track only
             POST /intelligence/finance   Finance track only
             POST /intelligence/security  Security track only
             GET  /reports/{id}           Retrieve a past report
             GET  /health                 Liveness probe

  Webhook    Critical security findings POST to any URL in the request body


BRIGHT DATA TOOLS USED
-----------------------
  - SERP API          Competitor news, buying signals, threat hunting,
                      compliance monitoring, brand exposure scanning
  - Web Unlocker      Competitor pricing pages, regulatory sites,
                      vendor security pages, threat source pages
  - Scraping Browser  JS-heavy sites (CISA, NIST, GDPR portals)
  - Web Scraper API   Indeed job postings dataset, LinkedIn company profiles


TECH STACK
----------
  - Python 3.14
  - FastAPI + Uvicorn (REST API and static file serving)
  - LangGraph (parallel multi-track orchestration)
  - Anthropic Claude API with prompt caching
  - Pydantic v2 (all data models and schemas)
  - httpx (async HTTP for Bright Data REST APIs)
  - Playwright via Bright Data Scraping Browser (JS rendering)


SETUP
-----
  1. Clone the repository
  2. Copy .env.example to .env and fill in your API keys:
       ANTHROPIC_API_KEY=...
       BRIGHTDATA_API_KEY=...
       BRIGHTDATA_CUSTOMER_ID=...
       BRIGHTDATA_UNLOCKER_ZONE=...
       BRIGHTDATA_SCRAPING_BROWSER_ZONE=...
       BRIGHTDATA_SCRAPING_BROWSER_PASSWORD=...
  3. pip install -r requirements.txt
  4. python -m uvicorn api.server:app --host 0.0.0.0 --port 8001 --reload
  5. Open http://localhost:8001


EXAMPLE REQUEST
---------------
  curl -X POST http://localhost:8001/intelligence \
    -H "Content-Type: application/json" \
    -d '{
      "target": "Salesforce",
      "tracks": ["gtm", "finance", "security"],
      "context": "enterprise CRM market",
      "webhook_url": "https://your-crm.com/webhook/alerts"
    }'
