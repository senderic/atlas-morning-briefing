
import os
import sys
import logging
from scripts.briefing_runner import load_config
from scripts.email_distributor import EmailDistributor
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def test_email():
    # Load environment variables
    load_dotenv(override=True)
    
    logger.info(f"DEBUG: RECIPIENT_EMAIL env var: {os.environ.get('RECIPIENT_EMAIL')}")
    
    config_path = "config.yaml"
    config = load_config(config_path)
    
    sender_email = os.environ.get("GMAIL_USER")
    sender_password = os.environ.get("GMAIL_APP_PASSWORD")
    
    if not sender_email or not sender_password:
        logger.error("GMAIL_USER or GMAIL_APP_PASSWORD not set in environment")
        return

    distributor = EmailDistributor(
        sender_email=sender_email,
        sender_password=sender_password
    )
    
    # Use existing files from today (Mar 25, 2026)
    markdown_path = "Atlas-Briefing-2026.03.25.md"
    pdf_path = "Atlas-Briefing-2026.03.25.pdf"
    
    if not os.path.exists(markdown_path):
        logger.error(f"Markdown file not found: {markdown_path}")
        return
        
    with open(markdown_path, "r") as f:
        markdown_content = f.read()
        
    logger.info(f"Starting test email distribution to: {config.get('email_recipients')}")
    
    results = distributor.distribute(
        config=config,
        markdown_content=markdown_content,
        pdf_path=pdf_path if os.path.exists(pdf_path) else None,
        subject="TEST: Atlas Morning Briefing - Multiple Recipients"
    )
    
    logger.info(f"Distribution results: {results}")

if __name__ == "__main__":
    test_email()
