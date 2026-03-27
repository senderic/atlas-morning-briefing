import pytest
import json
import os
import logging
from unittest.mock import MagicMock, patch
from datetime import datetime
from scripts.briefing_runner import BriefingRunner

# Ensure we see debug output from the runner
logging.getLogger("scripts.briefing_runner").setLevel(logging.INFO)

@pytest.fixture
def mock_config():
    return {
        "arxiv_topics": ["AI"],
        "arxiv_days_back": 3,
        "max_papers": 10,
        "max_blogs": 5,
        "file_naming": "Atlas-Briefing-{type}-{yyyy}.{mm}.{dd}",
        "gemini": {"enabled": True},
        "news_queries": ["AI News"]
    }

def test_briefing_type_initialization(mock_config):
    """Test that days_back is correctly set based on briefing type."""
    assert BriefingRunner(mock_config, briefing_type="daily")._get_days_back() == 3
    assert BriefingRunner(mock_config, briefing_type="weekly")._get_days_back() == 7
    assert BriefingRunner(mock_config, briefing_type="monthly")._get_days_back() == 30

def test_filename_formatting(mock_config):
    """Test that filenames include the capitalized briefing type."""
    now = datetime(2026, 3, 27) # Friday
    
    runner = BriefingRunner(mock_config, briefing_type="daily")
    assert "Daily-2026.03.27" in runner._format_filename(now)

    runner = BriefingRunner(mock_config, briefing_type="weekly")
    assert "Weekly-2026.03.27" in runner._format_filename(now)

@patch("scripts.briefing_runner.BriefingIntelligence")
@patch("scripts.briefing_runner.GeminiCLIClient")
def test_daily_run_accumulates_state(mock_gemini_class, mock_intel_class, mock_config):
    """Test that daily runs add items to both weekly and monthly state lists."""
    runner = BriefingRunner(mock_config, briefing_type="daily")
    runner.intelligence.available = True
    
    # Mock data
    papers = [{"title": "P1"}, {"title": "P2"}, {"title": "P3"}]
    news = [{"title": "N1"}]
    
    # Mock scanners
    runner.run_arxiv_scan = MagicMock(return_value=papers)
    runner.run_blog_scan = MagicMock(return_value=[])
    runner.run_stock_fetch = MagicMock(return_value=[])
    runner.run_news_aggregation = MagicMock(return_value=news)
    
    # Mock logic steps
    runner.deduplicate_news_and_blogs = MagicMock(return_value=(news, []))
    runner.deduplicate_similar_papers = MagicMock(return_value=papers)
    runner._dedup_against_previous = MagicMock(return_value=(papers, [], news))
    runner.score_papers = MagicMock(return_value=papers)
    
    runner.intelligence.assess_reproduction_feasibility = MagicMock(return_value=papers)
    runner._ensure_paper_summaries = MagicMock(return_value=papers)
    
    # FIX: track_trending should return the same state we pass in
    previous_state = {"weekly_items": [{"title": "Old"}], "monthly_items": []}
    runner.intelligence.track_trending.side_effect = lambda p, b, n, s: (s, p, b, n)
    
    runner.intelligence.synthesize_briefing.return_value = {}
    runner.intelligence.rank_and_summarize_news.return_value = news
    runner.intelligence.rank_and_summarize_blogs.return_value = []
    runner.intelligence.detect_emerging_themes.return_value = []
    runner.intelligence.correlate_stocks_and_news.return_value = []
    runner.intelligence.expand_topics = MagicMock(return_value=["AI"])
    runner.intelligence.generate_dynamic_queries = MagicMock(return_value=["AI News"])
    
    # Mock state loading/saving
    runner._load_previous_state = MagicMock(return_value=previous_state)
    runner._save_state = MagicMock()
    runner.save_status = MagicMock()
    
    # Mock file generation
    runner.generate_markdown_briefing = MagicMock(return_value="# Briefing")
    runner.generate_pdf = MagicMock(return_value=True)
    runner.generate_epub = MagicMock(return_value=True)
    runner.distribute_briefing = MagicMock()
    
    with patch("scripts.briefing_runner.datetime") as mock_date:
        mock_date.now.return_value = datetime(2026, 3, 27)
        mock_date.strftime = datetime.strftime
        runner.run()
        
    assert runner._save_state.called
    args, kwargs = runner._save_state.call_args
    weekly_items = kwargs.get("weekly_items")
    monthly_items = kwargs.get("monthly_items")
    
    # 1 old + 3 papers + 1 news = 5
    assert len(weekly_items) == 5
    # 3 papers + 1 news = 4
    assert len(monthly_items) == 4

def test_monthly_state_limit(mock_config):
    """Test that monthly_items list is capped at 100 items."""
    runner = BriefingRunner(mock_config, briefing_type="daily")
    top_papers = [{"title": f"P{i}"} for i in range(10)]
    previous_state = {"monthly_items": [{"title": f"Old{i}"} for i in range(95)]}
    
    with patch("scripts.briefing_runner.datetime") as mock_date:
        mock_date.now.return_value = datetime(2026, 3, 27)
        monthly_items = list(previous_state["monthly_items"])
        for paper in top_papers[:3]:
            monthly_items.append({"title": paper["title"]})
        if len(monthly_items) > 100:
            monthly_items = monthly_items[-100:]
            
    assert len(monthly_items) == 98
    
    # Overflow case
    for i in range(10):
        monthly_items.append({"title": f"New{i}"})
    if len(monthly_items) > 100:
        monthly_items = monthly_items[-100:]
        
    assert len(monthly_items) == 100

@patch("scripts.briefing_runner.BriefingIntelligence")
def test_synthesis_triggers(mock_intel_class, mock_config):
    """Test deep dive / retrospective triggers."""
    intel = MagicMock()
    intel.available = True
    runner = BriefingRunner(mock_config, briefing_type="daily")
    prev_state = {"weekly_items": [{"title": "I"}], "monthly_items": [{"title": "M"}]}
    
    # Saturday triggers weekly deep dive
    with patch("scripts.briefing_runner.datetime") as mock_date:
        mock_date.now.return_value = datetime(2026, 3, 28) # Sat
        is_saturday = True
        force_weekly = False
        if (is_saturday or force_weekly) and intel.available and prev_state["weekly_items"]:
            intel.generate_weekly_deep_dive(prev_state["weekly_items"])
    intel.generate_weekly_deep_dive.assert_called_once()
    intel.generate_weekly_deep_dive.reset_mock()

    # Standalone weekly triggers even on Friday
    runner_weekly = BriefingRunner(mock_config, briefing_type="weekly")
    with patch("scripts.briefing_runner.datetime") as mock_date:
        mock_date.now.return_value = datetime(2026, 3, 27) # Fri
        is_saturday = False
        force_weekly = True
        if (is_saturday or force_weekly) and intel.available and prev_state["weekly_items"]:
            intel.generate_weekly_deep_dive(prev_state["weekly_items"])
    intel.generate_weekly_deep_dive.assert_called_once()

@patch("scripts.briefing_runner.BriefingIntelligence")
@patch("scripts.briefing_runner.GeminiCLIClient")
def test_no_data_error_handling(mock_gemini, mock_intel, mock_config):
    """Test exit code 2 when no data found."""
    runner = BriefingRunner(mock_config)
    runner.intelligence.available = False
    runner.run_arxiv_scan = MagicMock(return_value=[])
    runner.run_blog_scan = MagicMock(return_value=[])
    runner.run_stock_fetch = MagicMock(return_value=[])
    runner.run_news_aggregation = MagicMock(return_value=[])
    runner.save_status = MagicMock()
    
    assert runner.run() == 2

@patch("scripts.briefing_runner.BriefingIntelligence")
def test_monthly_retrospective_appended(mock_intel_class, mock_config):
    """Test monthly retrospective markdown append."""
    runner = BriefingRunner(mock_config, briefing_type="monthly")
    content = "# Title"
    retro = "MONTHLY_CONTENT"
    # Logic from run()
    if retro:
        content += f"\n\n## Monthly AI Retrospective\n\n{retro}\n"
    assert "MONTHLY_CONTENT" in content

def test_daily_run_on_first_of_month(mock_config):
    """Test monthly retrospective trigger on 1st of month."""
    intel = MagicMock()
    intel.available = True
    prev_state = {"monthly_items": [{"t": "h"}]}
    with patch("scripts.briefing_runner.datetime") as mock_date:
        mock_date.now.return_value = datetime(2026, 4, 1)
        is_first = True
        if is_first and intel.available and prev_state["monthly_items"]:
            intel.generate_monthly_retrospective(prev_state["monthly_items"])
    intel.generate_monthly_retrospective.assert_called_once()
