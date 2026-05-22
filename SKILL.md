---
name: morning-briefing
description: Generate a daily AI research + market + news briefing. Use when setting up automated morning briefings, research digests, or daily knowledge feeds. Covers arxiv papers, tech blogs, stock watchlist, industry news, and paper recommendations. Outputs Kindle PDF + channel message. Configurable topics, sources, stocks, and delivery schedule. Optionally uses Gemini CLI for AI-powered synthesis and summarization.
inputs:
  config_path:
    type: string
    required: true
    description: Path to configuration YAML file
  dry_run:
    type: boolean
    default: false
    description: Generate briefing without sending email
outputs:
  pdf_path:
    type: string
    description: Path to generated PDF file
  markdown_path:
    type: string
    description: Path to generated markdown file
  status:
    type: object
    description: Run status (papers_found, blogs_found, stocks_fetched, news_found, errors, elapsed_seconds)
triggers:
  schedule: "50 6 * * *"
  manual: "generate my morning briefing"
requires:
  python: ">=3.10"
  env:
    required:
      - FINNHUB_API_KEY
      - BRAVE_API_KEY
    optional:
      - GMAIL_USER
      - GMAIL_APP_PASSWORD
---

# Morning Briefing Skill

Generates a comprehensive morning briefing covering AI/ML papers (arxiv), tech blogs, stock watchlist, industry news, and paper recommendations for reproduction. Outputs as Kindle-optimized PDF with optional email delivery.

Enhanced with **Gemini CLI** for intelligent summarization, cross-section synthesis, and semantic paper scoring. Falls back gracefully to deterministic mode when Gemini CLI is unavailable.

## Prerequisites

### Required
- **Python 3.10+** (`python3 --version`)
- **pip** (`pip3 --version`)
- **Gemini CLI** (`gemini --version`)

### API Keys (all free tier)
| Service | Purpose | Sign Up | Free Tier |
|---------|---------|---------|-----------|
| **Finnhub** | Stock market data | [finnhub.io](https://finnhub.io/) | 60 calls/min |
| **Brave Search** | News aggregation | [brave.com/search/api](https://brave.com/search/api/) | 2000 queries/mo |
| **Gmail App Password** | Kindle email delivery | [myaccount.google.com](https://myaccount.google.com/apppasswords) | Free |

### Optional
- **Kindle Scribe/device** -- For PDF delivery via email. See `references/kindle_setup.md`
- **CJK fonts** -- For Chinese/Japanese/Korean support in PDFs:
  ```bash
  # Ubuntu/Debian
  sudo apt install fonts-noto-cjk
  # macOS
  brew install font-noto-sans-cjk
  ```

### System Dependencies
```bash
# Ubuntu/Debian
sudo apt install python3-venv python3-pip

# macOS
brew install python3
```

## Features

### Deterministic (no LLM required)
- **ArXiv Paper Scanning**: Tracks new papers on configured topics
- **Blog Feed Monitoring**: Aggregates updates from RSS feeds
- **Stock Watchlist**: Fetches market data for configured tickers (Finnhub API)
- **News Aggregation**: Collects top AI/tech headlines (Brave Search API)
- **Paper Scoring**: Ranks papers by reproduction value (code availability, topic match, recency)
- **Cross-Section Deduplication**: Removes duplicate content between news and blogs
- **Kindle-Optimized PDF**: 6x8 inch format with CJK support
- **Email Delivery**: Send directly to Kindle via SMTP
- **Config Validation**: Catches configuration errors at startup
- **Status Reporting**: Generates status.json for monitoring

### Intelligence Layer (Gemini CLI, optional)
- **Topic Expansion**: Suggests related search queries using Gemini 2.5 Flash Lite
- **Paper Summarization**: 1-2 sentence takeaways for each paper using Gemini 3 Flash Preview
- **Semantic Scoring**: Relevance scoring using LLM understanding (beyond TF-IDF)
- **Stock-News Correlation**: Links stock movements to news drivers
- **Reproduction Assessment**: Evaluates compute, data, and feasibility for top papers
- **Cross-Section Synthesis**: Finds themes across papers, news, and blogs
- **Editorial Intro**: Opens briefing with today's key insight
- **Market Trend Summary**: 2-sentence market analysis with key drivers

## Setup

### 1. Install Dependencies

It's recommended to use a virtual environment:

```bash
cd atlas-morning-briefing
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure API Keys

Set environment variables:

```bash
export FINNHUB_API_KEY="your_finnhub_key"
export BRAVE_API_KEY="your_brave_search_key"
export GMAIL_USER="your_email@gmail.com"
export GMAIL_APP_PASSWORD="your_app_password"
```

### 3. Configure Topics and Sources

Edit `config.yaml`:

```yaml
arxiv_topics:
  - "Agent Evaluation"
  - "Multi-Agent Systems"

blog_feeds:
  - name: "Anthropic"
    url: "https://www.anthropic.com/rss.xml"

stocks:
  - AMZN
  - GOOGL

news_queries:
  - "AI artificial intelligence"

kindle_email: "YOUR_NAME@kindle.com"
sender_email: "YOUR_EMAIL@gmail.com"

gemini:
  enabled: true
  models:
    heavy: "pro"
    medium: "flash"
    light: "flash-lite"
```

See `references/config_guide.md` for full configuration options.

### 4. Set Up Kindle Email Delivery

See `references/kindle_setup.md` for instructions on configuring your Kindle email address.

## Usage

### Generate Briefing (Dry Run)

```bash
python3 scripts/briefing_runner.py --config config.yaml --dry-run
```

### Generate and Send to Kindle

```bash
python3 scripts/briefing_runner.py --config config.yaml
```

## Run Status

After each run, a `status.json` file is generated:

```json
{
  "timestamp": "2026-03-06T06:50:12",
  "papers_found": 14,
  "blogs_found": 3,
  "stocks_fetched": 5,
  "news_found": 8,
  "intelligence_enabled": true,
  "errors": [],
  "pdf_generated": true,
  "email_sent": true,
  "elapsed_seconds": 7.2
}
```

## Scheduling

Set up a daily cron job:

```bash
crontab -e
```

Add:

```
0 7 * * * /path/to/atlas-morning-briefing/run_briefing.sh >> /path/to/atlas-morning-briefing/logs/briefing.log 2>&1
```

Create wrapper script (`run_briefing.sh`):

```bash
#!/bin/bash
cd /path/to/atlas-morning-briefing
source venv/bin/activate
source .env  # Load API keys
python3 scripts/briefing_runner.py --config config.yaml
```

## Cost Estimate

### Total Cost: $0.00/month

All external APIs (ArXiv, Finnhub, Brave, Gmail) have free tiers sufficient for daily use. Gemini CLI is also free to use.

## Paper Scoring Criteria

Papers are scored based on:

- **has_code** (weight: 5): Links to open source code repository
- **topic_match** (weight: 3): Cosine similarity to configured topics (TF-IDF)
- **semantic_score** (Gemini): LLM-assessed relevance with explanation
- **recency** (weight: 2): Days since publication
- **citation_count** (weight: 1): Number of citations (if available)

Reproduction difficulty is estimated as S/M/L/XL based on:
- Dependencies complexity
- Dataset size
- Compute requirements

When Gemini is enabled, reproduction assessment includes specific compute estimates and blocker identification.

## Troubleshooting

### No papers found
- Check arxiv_topics in config.yaml match arxiv categories
- Verify date range is not too narrow

### PDF generation fails
- PDF generation is disabled by default; enable via `pdf.enabled: true` in config.yaml
- Ensure fonts are installed for CJK support
- Check markdown formatting is valid

### Email delivery fails
- Verify GMAIL_USER and GMAIL_APP_PASSWORD are set
- Check sender_email matches GMAIL_USER
- Ensure Kindle email is whitelisted in Amazon account

### API rate limits
- Finnhub: Free tier allows 60 calls/minute
- Brave Search: Check your plan limits
- Rate limiting is built-in (0.5s delay between Finnhub calls, 1.0s between Brave calls)

### LLM call budget exhausted
- The intelligence layer has a per-run call budget (default: 50 calls for Gemini)
- With all features enabled and many papers, this can be exceeded
- Increase via `gemini.max_calls_per_run` in config.yaml:
  ```yaml
  gemini:
    max_calls_per_run: 80
  ```
- Typical usage: ~20-30 calls with default settings

### Gemini CLI errors
- Verify the `gemini` binary is on PATH (`which gemini`)
- Ensure GEMINI_API_KEY (or GEMINI_API_KEY_* for rotation) is set
- Set `gemini.enabled: false` to disable and run deterministically

### Config validation errors
- The runner validates config at startup and reports specific errors
- Check the error message for the invalid field and expected type

## Architecture

```
briefing_runner.py (orchestrator)
├── [Intelligence] Topic expansion (Gemini Light)
├── arxiv_scanner.py → papers
├── blog_scanner.py → blogs
├── stock_fetcher.py → stocks
├── news_aggregator.py → news
├── [Dedup] Cross-section deduplication
├── [Intelligence] Paper summarization (Gemini Medium)
├── [Intelligence] Semantic scoring (Gemini Medium)
├── [Intelligence] Stock-news correlation (Gemini Heavy)
├── paper_scorer.py → scored papers
├── [Intelligence] Reproduction assessment (Gemini Medium)
├── [Intelligence] Cross-section synthesis (Gemini Heavy)
├── [Generate markdown briefing with editorial content]
├── pdf_generator.py → Atlas-Briefing.pdf
├── email_distributor.py → Email to Kindle
└── [Save status.json]
```

## Exit Codes

- **0**: Success
- **1**: Partial failure (some scanners failed but briefing generated)
- **2**: Total failure (unable to generate briefing or invalid config)

## References

- `references/config_guide.md`: Full configuration reference
- `references/kindle_setup.md`: Kindle email setup instructions
- `examples/sample-briefing.md`: Example generated briefing
