"""
Kalshi Events Scraper
Optimized for scraping settled (historical) events with multi-threading.

Usage:
    python scraper.py                                              # Settled events (auto-filtered by actual_status)
    python scraper.py --min-dollar-volume 100000                   # Settled events with $100K+ volume
    python scraper.py --min-dollar-volume 1000000                  # Settled events with $1M+ volume
    python scraper.py --status open closed settled --filter-actual-status settled  # Query all, keep only settled

Output:
    kalshi_events_YYYYMMDD_HHMMSS.csv

Note:
    The 'volume' field from Kalshi API is already in dollars (not contracts).
    Default status query is 'settled' for historical analysis.

    The API sometimes returns events with wrong status. This scraper now AUTOMATICALLY
    filters by actual_status (from market data) when querying a single status to ensure
    you only get truly settled/closed/open markets. Use --filter-actual-status to override.
"""

import argparse
import requests
import csv
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from threading import Lock

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2/events"

# Thread-safe print lock
print_lock = Lock()

def safe_print(msg):
    with print_lock:
        print(msg)


def get_session() -> requests.Session:
    """Create a session with retry logic."""
    session = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504]
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=10, pool_maxsize=10)
    session.mount("https://", adapter)
    return session


def fetch_events_page(session: requests.Session, status: str, cursor: str = None) -> dict:
    """Fetch a single page of events with nested market data."""
    params = {
        "limit": 200,
        "status": status,
        "with_nested_markets": "true"
    }
    if cursor:
        params["cursor"] = cursor
    
    response = session.get(BASE_URL, params=params, timeout=60)
    response.raise_for_status()
    return response.json()


def calculate_event_dollar_volume(event: dict) -> int:
    """Calculate total dollar volume across all markets in an event."""
    markets = event.get("markets", [])
    return sum(m.get("volume", 0) for m in markets)


def calculate_event_open_interest(event: dict) -> int:
    """Calculate total open interest across all markets in an event."""
    markets = event.get("markets", [])
    return sum(m.get("open_interest", 0) for m in markets)


def get_event_close_time(event: dict) -> str:
    """Get the latest close time from markets in an event."""
    markets = event.get("markets", [])
    close_times = [m.get("close_time", "") for m in markets if m.get("close_time")]
    return max(close_times) if close_times else ""


def get_market_count(event: dict) -> int:
    """Get the number of markets in an event."""
    return len(event.get("markets", []))


def get_actual_event_status(event: dict) -> str:
    """
    Determine actual event status from its markets' statuses.
    Don't trust the query parameter - check the actual market data.
    """
    markets = event.get("markets", [])
    
    if not markets:
        return "unknown"
    
    market_statuses = [m.get("status", "").lower() for m in markets]
    
    # If all markets are settled/finalized, event is settled
    settled_statuses = {"settled", "finalized", "determined"}
    if all(s in settled_statuses for s in market_statuses):
        return "settled"
    
    # If any market is active, event is open
    if any(s == "active" for s in market_statuses):
        return "open"
    
    # If any market is closed (but not settled), event is closed
    if any(s == "closed" for s in market_statuses):
        return "closed"
    
    # Default fallback
    return "unknown"


def process_event(event: dict, min_dollar_volume: int) -> dict:
    """Process a single event and return enriched data if it passes filters."""
    dollar_volume = calculate_event_dollar_volume(event)
    
    if dollar_volume < min_dollar_volume:
        return None
    
    # Get actual status from market data, not from query parameter
    actual_status = get_actual_event_status(event)
    
    return {
        "event_ticker": event.get("event_ticker"),
        "series_ticker": event.get("series_ticker"),
        "title": event.get("title"),
        "sub_title": event.get("sub_title"),
        "category": event.get("category"),
        "mutually_exclusive": event.get("mutually_exclusive"),
        "dollar_volume": dollar_volume,
        "open_interest": calculate_event_open_interest(event),
        "market_count": get_market_count(event),
        "close_time": get_event_close_time(event),
        "actual_status": actual_status,  # True status from market data
    }


def fetch_all_events_for_status(status: str, min_dollar_volume: int = 0) -> list:
    """Fetch all events for a status with pagination."""
    session = get_session()
    all_events = []
    filtered_count = 0
    cursor = None
    page = 0
    
    safe_print(f"  [{status}] Starting fetch...")
    
    while True:
        page += 1
        try:
            data = fetch_events_page(session, status, cursor)
            events = data.get("events", [])
            
            if not events:
                break
            
            for event in events:
                processed = process_event(event, min_dollar_volume)
                if processed:
                    all_events.append(processed)
                else:
                    filtered_count += 1
            
            if page % 25 == 0:
                safe_print(f"  [{status}] Page {page}: {len(all_events)} events kept, {filtered_count} filtered")
            
            cursor = data.get("cursor")
            if not cursor:
                break
            
            time.sleep(0.05)
            
        except Exception as e:
            safe_print(f"  [{status}] Error on page {page}: {e}")
            break
    
    safe_print(f"  [{status}] Complete: {len(all_events)} events (filtered {filtered_count})")
    return all_events


def scrape_all_events(statuses: list, min_dollar_volume: int = 0, filter_actual_status: str = None) -> list:
    """Scrape events from multiple statuses in parallel."""
    all_events = []
    
    print(f"Starting scrape at {datetime.now().isoformat()}")
    print(f"Filters: query_statuses={statuses}, min_dollar_volume=${min_dollar_volume:,}")
    if filter_actual_status:
        print(f"         filter_actual_status={filter_actual_status}")
    print("-" * 60)
    
    # Fetch different statuses in parallel
    with ThreadPoolExecutor(max_workers=len(statuses)) as executor:
        future_to_status = {
            executor.submit(fetch_all_events_for_status, status, min_dollar_volume): status 
            for status in statuses
        }
        
        for future in as_completed(future_to_status):
            status = future_to_status[future]
            try:
                events = future.result()
                all_events.extend(events)
            except Exception as e:
                print(f"  [{status}] Failed: {e}")
    
    # Deduplicate by event_ticker
    seen = set()
    unique_events = []
    for event in all_events:
        ticker = event.get("event_ticker")
        if ticker and ticker not in seen:
            seen.add(ticker)
            unique_events.append(event)

    # Auto-filter by actual status to match query status (if single status queried)
    # This ensures we only get truly settled/closed/open markets, not what API claims
    if filter_actual_status is None and len(statuses) == 1:
        filter_actual_status = statuses[0]
        print(f"Auto-filtering to actual_status='{filter_actual_status}' (matches query status)")

    # Filter by actual status if requested or auto-determined
    if filter_actual_status:
        before_count = len(unique_events)
        unique_events = [e for e in unique_events if e.get("actual_status") == filter_actual_status]
        filtered_count = before_count - len(unique_events)
        if filtered_count > 0:
            print(f"Filtered out {filtered_count} events with mismatched actual_status")
        print(f"Kept {len(unique_events)} events with actual_status='{filter_actual_status}'")
    
    print("-" * 60)
    print(f"Total unique events: {len(unique_events)}")
    
    return unique_events


def events_to_csv(events: list, filename: str):
    """Save events to CSV file."""
    if not events:
        print("No events to save.")
        return
    
    columns = [
        "event_ticker",
        "series_ticker", 
        "title",
        "sub_title",
        "category",
        "mutually_exclusive",
        "dollar_volume",
        "open_interest",
        "market_count",
        "close_time",
        "actual_status",  # True status from market data
    ]
    
    # Sort by dollar volume descending
    events.sort(key=lambda x: x.get("dollar_volume", 0), reverse=True)
    
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(events)
    
    print(f"Saved {len(events)} events to {filename}")


def main():
    parser = argparse.ArgumentParser(description="Scrape Kalshi events to CSV")
    parser.add_argument(
        "--min-dollar-volume",
        type=int,
        default=0,
        help="Minimum dollar volume (default: 0). Use 1000000 for $1M+"
    )
    parser.add_argument(
        "--status",
        type=str,
        nargs="+",
        default=["settled"],  # Default to settled for historical analysis
        choices=["open", "closed", "settled"],
        help="Event statuses to query from API (default: settled)"
    )
    parser.add_argument(
        "--filter-actual-status",
        type=str,
        default=None,
        choices=["open", "closed", "settled"],
        help="Filter results by actual status from market data (optional)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output filename (default: kalshi_events_TIMESTAMP.csv)"
    )
    args = parser.parse_args()
    
    # Scrape events
    events = scrape_all_events(args.status, args.min_dollar_volume, args.filter_actual_status)
    
    # Save to CSV
    if args.output:
        filename = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"market_data/kalshi_events_{timestamp}.csv"
    
    events_to_csv(events, filename)
    
    # Print summaries
    if events:
        print("\n" + "=" * 60)
        print("SUMMARY BY CATEGORY")
        print("=" * 60)
        
        categories = {}
        for event in events:
            cat = event.get("category", "Unknown")
            categories[cat] = categories.get(cat, 0) + 1
        
        for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
            print(f"  {cat}: {count}")
        
        print("\n" + "=" * 60)
        print("SUMMARY BY STATUS (actual from market data)")
        print("=" * 60)
        
        statuses = {}
        for event in events:
            status = event.get("actual_status", "Unknown")
            statuses[status] = statuses.get(status, 0) + 1
        
        for status, count in sorted(statuses.items(), key=lambda x: -x[1]):
            print(f"  {status}: {count}")
        
        print("\n" + "=" * 60)
        print("VOLUME STATISTICS")
        print("=" * 60)
        
        volumes = [e.get("dollar_volume", 0) for e in events]
        print(f"  Total dollar volume: ${sum(volumes):,}")
        print(f"  Mean dollar volume:  ${sum(volumes) // len(volumes):,}")
        print(f"  Max dollar volume:   ${max(volumes):,}")
        print(f"  Min dollar volume:   ${min(volumes):,}")
        print(f"  Events $1M+:         {sum(1 for v in volumes if v >= 1_000_000)}")
        print(f"  Events $100K+:       {sum(1 for v in volumes if v >= 100_000)}")
    
    return filename


if __name__ == "__main__":
    output_file = main()
    print(f"\nDone! Output: {output_file}")