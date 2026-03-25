import os
import sys
from datetime import datetime
from pathlib import Path

# Ensure scripts directory is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from scripts.briefing_runner import BriefingRunner, load_config

def run_mock_test():
    # Load real config to get distribution settings, but we will override data
    config = load_config("config.yaml")
    
    # Mock data
    stocks = [
        {"symbol": "NVDA", "current_price": 142.80, "percent_change": -3.20, "news_correlation": "US tightens AI chip export controls"},
        {"symbol": "PLTR", "current_price": 35.50, "percent_change": 5.40, "news_correlation": "New DoD contract announced"}
    ]
    
    news = [
        {
            "title": "US Tightens AI Chip Export Controls to Southeast Asia",
            "url": "https://example.com/reuters-export",
            "brief_summary": "The US government has expanded export restrictions on high-end AI chips to more countries in Southeast Asia to prevent unauthorized re-exports."
        }
    ]
    
    blogs = [
        {
            "title": "How Claude Uses Tools in Production",
            "source": "Anthropic",
            "link": "https://www.anthropic.com/blog/tool-use",
            "score_combined": 5,
            "brief_summary": "A deep dive into the tool-use architecture powering Claude's real-world capabilities."
        }
    ]
    
    top_papers = [
        {
            "title": "AgentBench v2: A Comprehensive Benchmark for Multi-Agent Evaluation",
            "authors": ["Chen", "Wang", "Li"],
            "arxiv_url": "http://arxiv.org/abs/2403.00001",
            "brief_summary": "This paper introduces AgentBench v2 with 12 new evaluation tasks covering collaboration, competition, and tool-use scenarios.",
            "score_combined": 5,
            "repro_total": 22,
            "repro_verdict": "Highly reproducible",
            "reproduction_difficulty": "Medium"
        }
    ]
    
    papers = top_papers # Just for simple test
    
    synthesis = {
        "editorial_intro": "Today's briefing highlights a surge of interest in multi-agent evaluation frameworks. NVIDIA's dip correlates with tightened export controls."
    }
    
    runner = BriefingRunner(config)
    
    print("Generating Mock Briefing...")
    markdown_content = runner.generate_markdown_briefing(
        papers=papers,
        blogs=blogs,
        stocks=stocks,
        news=news,
        top_papers=top_papers,
        synthesis=synthesis,
        market_trend="Markets are mixed today with tech facing some headwinds from export news."
    )
    
    print("\n--- BEGIN MOCK MARKDOWN ---")
    print(markdown_content[:500] + "...")
    print("--- END MOCK MARKDOWN ---\n")
    
    # Check if title and timestamp are present
    if "# Atlas Morning Briefing" in markdown_content and "|" in markdown_content:
        print("✅ SUCCESS: Title and Timestamp found in markdown.")
    else:
        print("❌ FAILURE: Title or Timestamp missing!")
        sys.exit(1)

    # Generate files
    filename = "Mock-Briefing-Test"
    pdf_path = f"{filename}.pdf"
    epub_path = f"{filename}.epub"
    
    runner.generate_pdf(markdown_content, pdf_path)
    runner.generate_epub(markdown_content, epub_path)
    
    print(f"Generated {pdf_path} and {epub_path}")

    # Send email if requested
    if os.environ.get("GMAIL_USER") and os.environ.get("GMAIL_APP_PASSWORD"):
        print("Sending sample email...")
        runner.distribute_briefing(markdown_content, pdf_path, "Atlas Sample Briefing (Mock Test)", epub_path=epub_path)
        print("Email distribution triggered.")
    else:
        print("Skipping email send (credentials not in environment).")

if __name__ == "__main__":
    run_mock_test()
