import os
import re
import json
from datetime import datetime

STATE_FILE = ".atlas-state.json"

def parse_briefing(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract date from filename
    date_match = re.search(r"(\d{4})\.(\d{2})\.(\d{2})", file_path)
    if not date_match:
        return []
    
    date_str = f"{date_match.group(1)}-{date_match.group(2)}-{date_match.group(3)}"
    items = []
    
    # 1. Parse Top Papers
    paper_section = re.search(r"## Top Papers(.*?)(##|$)", content, re.DOTALL)
    if paper_section:
        papers = re.findall(r"### \d+\. \[(.*?)\]", paper_section.group(1))
        for title in papers[:3]:
            items.append({"date": date_str, "type": "paper", "title": title})
            
    # 2. Parse AI & Tech News
    news_section = re.search(r"## AI & Tech News(.*?)(##|$)", content, re.DOTALL)
    if news_section:
        # Match **[Title](url)** or **Title** (in case no URL)
        news = re.findall(r"\*\*\[?(.*?)\]?(?:\(http.*?\))?\*\*", news_section.group(1))
        for title in news[:3]:
            items.append({"date": date_str, "type": "news", "title": title})

    # 3. Fallback: Parse Blog Updates if section exists and we need more items
    blog_section = re.search(r"## Blog Updates(.*?)(##|$)", content, re.DOTALL)
    if blog_section and len(items) < 3:
        blogs = re.findall(r"\*\*\[?(.*?)\]?(?:\(http.*?\))?\*\*", blog_section.group(1))
        for title in blogs[:3]:
            items.append({"date": date_str, "type": "blog", "title": title})
            
    return items

def backfill():
    all_items = []
    files = [f for f in os.listdir('.') if f.startswith("Atlas-Briefing-") and f.endswith(".md")]
    files.sort()
    
    for f in files:
        print(f"Parsing {f}...")
        parsed = parse_briefing(f)
        print(f"  Found {len(parsed)} items")
        all_items.extend(parsed)
        
    if not all_items:
        print("No items found to backfill.")
        return

    state = {}
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'r') as f:
                state = json.load(f)
        except:
            state = {}
            
    # Use sets to avoid exact duplicates
    seen_titles = set()
    unique_items = []
    for item in all_items:
        if item["title"] not in seen_titles:
            unique_items.append(item)
            seen_titles.add(item["title"])

    state["weekly_items"] = unique_items[-42:]
    state["monthly_items"] = unique_items[-100:]
    
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)
        
    print(f"Successfully backfilled {len(unique_items)} unique items into {STATE_FILE}")

if __name__ == "__main__":
    backfill()
