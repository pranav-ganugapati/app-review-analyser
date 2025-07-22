import streamlit as st
import os
import json
import time
import re
from datetime import datetime, timedelta
from collections import defaultdict

# Third-party libraries
from google_play_scraper import reviews as google_reviews, Sort as GoogleSort
from app_store_scraper import AppStore
import google.generativeai as genai
from supabase import create_client
from dotenv import load_dotenv

# --- CONFIGURATION ---
load_dotenv()
SCRAPE_DAYS = 90
MAX_REVIEWS_PER_SOURCE = 1000
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GENERATIVE_MODEL = "gemini-1.5-pro-latest"
API_CALL_DELAY = 1
TOPIC_LISTS = {
    "grocery": ["Delivery Experience", "Product Quality", "Product Availability", "Pricing & Value", "App Experience (UI/UX)", "Payments & Refunds", "Customer Support", "Offers & Discounts", "Order Accuracy & Packaging", "Miscellaneous"],
    "games": ["Game Experience & Variety", "Trust & Fair Play", "Winning & Payouts", "Payments & Withdrawals", "Rewards & Bonuses", "Account & Verification", "App Performance & UI", "Customer Support", "Ads & Promotions", "Overall Positive Experience", "Overall Negative Experience", "Miscellaneous"]
}
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
PROCESSING_BATCH_SIZE = 50

# Global logger to capture terminal-like output
LOGS = []
LOG_AREA = None

def log(message):
    """Prints a message to the console and displays it in the Streamlit UI."""
    print(message)  # Keep printing to the actual terminal for debugging
    LOGS.append(str(message))
    if LOG_AREA:
        LOG_AREA.code("\n".join(LOGS))

# --- HELPER & CORE LOGIC FUNCTIONS (Using the new logger) ---

def get_supabase_client():
    if not SUPABASE_URL or not SUPABASE_KEY:
        log("❌ ERROR: Supabase credentials not set.")
        return None
    log("Connecting to Supabase...")
    try:
        client = create_client(SUPABASE_URL, SUPABASE_KEY)
        log("✅ Supabase connection successful.")
        return client
    except Exception as e:
        log(f"❌ Supabase connection error: {e}")
        return None

def scrape_play_store(app_id, cutoff_date):
    log(f"🚀 Scraping Google Play for '{app_id}'...")
    all_reviews, processed_count, token = [], 0, None
    while processed_count < MAX_REVIEWS_PER_SOURCE:
        try:
            result, token = google_reviews(app_id, lang='en', country='us', sort=GoogleSort.NEWEST, count=200, continuation_token=token)
            if not result or not token: break
            for r in result:
                if r['at'] < cutoff_date or processed_count >= MAX_REVIEWS_PER_SOURCE:
                    token = None
                    break
                processed_count += 1
                all_reviews.append({'id': r['reviewId'], 'store': 'Google Play', 'appid': app_id, 'Username': r['userName'], 'Date': r['at'].strftime('%Y-%m-%d'), 'Rating': r['score'], 'Review Text': r['content'], 'URL': r.get('url', f"https://play.google.com/store/apps/details?id={app_id}&reviewId={r['reviewId']}"), 'Timestamp': r['at'].isoformat(), 'Topic': 'N/A', 'Sentiment': 'N/A'})
            if not token: break
        except Exception as e:
            log(f"❌ Google Play scraping error: {e}")
            break
    log(f"✅ Found {len(all_reviews)} new reviews from Google Play.")
    return all_reviews

def scrape_app_store(app_name, country, cutoff_date):
    log(f"🍎 Scraping App Store for '{app_name}' in '{country}'...")
    try:
        scraper = AppStore(country=country, app_name=app_name)
        scraper.review(how_many=MAX_REVIEWS_PER_SOURCE)
        all_reviews = []
        for r in scraper.reviews:
            review_date = r['date']
            if review_date < cutoff_date: break
            all_reviews.append({'id': str(r['review_id']), 'store': 'App Store', 'appid': app_name, 'Username': r['userName'], 'Date': review_date.strftime('%Y-%m-%d'), 'Rating': r['rating'], 'Review Text': r['review'], 'URL': '', 'Timestamp': review_date.isoformat(), 'Topic': 'N/A', 'Sentiment': 'N/A'})
        log(f"✅ Found {len(all_reviews)} new reviews from App Store.")
        return all_reviews
    except Exception as e:
        log(f"⚠️ App Store scraping failed: {e}. Continuing without App Store reviews.")
        return []

def analyze_reviews_with_llm(model, review_list, topic_list):
    log(f"\n🤖 Starting classification for a batch of {len(review_list)} reviews...")
    analyzed_reviews = []
    for i, review in enumerate(review_list):
        progress_message = f"  ...classifying review {i+1}/{len(review_list)}... "
        if not review.get('Review Text', '').strip():
            log(progress_message + "Skipped (empty).")
            analyzed_reviews.append(review)
            continue
        
        prompt = f"""Analyze the following user review. You MUST categorize this review into one of the following topics:\n{topic_list}\n\nReview text:\n"{review['Review Text']}"\n\nInstructions:\n1. Choose the single most relevant topic from the provided list.\n2. If no topic is a good fit, you MUST use "Miscellaneous".\n3. Your response MUST be a JSON object with two keys: "sentiment" and "topic".\n4. The "sentiment" value must be one of: "Positive", "Negative", or "Neutral".\n5. The "topic" value MUST be one of the topics from the list."""
        try:
            response_text = model.generate_content(prompt).text
            match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if not match: raise ValueError("No valid JSON object found in LLM response.")
            json_string = match.group(0)
            result = json.loads(json_string)
            topic = result.get('topic', 'Miscellaneous')
            if topic not in topic_list: topic = 'Miscellaneous'
            review['Sentiment'] = result.get('sentiment', 'Error')
            review['Topic'] = topic
            log(progress_message + f"Sentiment: {review['Sentiment']}, Topic: {review['Topic']}")
            analyzed_reviews.append(review)
        except Exception as e:
            log(progress_message + f"Failed ({e}).")
            review['Sentiment'] = 'Error'
            review['Topic'] = 'Miscellaneous'
            analyzed_reviews.append(review)
        time.sleep(API_CALL_DELAY)
    return analyzed_reviews

def sync_raw_reviews_to_supabase(supabase, review_list):
    if not review_list: return 0
    log(f"  ...syncing {len(review_list)} raw reviews to Supabase...")
    records = [{'id': r['id'], 'store': r['store'], 'app_id': r['appid'], 'user_name': r['Username'], 'review_date': r['Date'], 'review_time': datetime.fromisoformat(r['Timestamp']).strftime('%H:%M:%S'), 'rating': r['Rating'], 'review_text': r['Review Text'], 'source_url': r['URL'], 'created_at': r['Timestamp'], 'topic': r['Topic'], 'sentiment': r['Sentiment']} for r in review_list]
    try:
        response = supabase.table('raw_reviews').upsert(records).execute()
        return len(response.data)
    except Exception as e:
        log(f"  ❌ Supabase raw sync error: {e}")
        return 0

def run_pipeline(industry, google_app_id, apple_app_name, apple_app_country):
    """This is the main pipeline function, adapted from your original `main`."""
    log("-" * 50 + f"\nStarting pipeline for '{industry}' industry...\n" + "-" * 50)
    
    topic_list = TOPIC_LISTS.get(industry)
    if not topic_list:
        log(f"❌ Error: Invalid industry '{industry}'. Please choose from {list(TOPIC_LISTS.keys())}.")
        return

    primary_app_id = google_app_id or apple_app_name
    if not primary_app_id:
        log("❌ No app ID or name provided. Exiting.")
        return

    cutoff_date = datetime.now() - timedelta(days=SCRAPE_DAYS)
    all_scraped_reviews = []
    
    try:
        if google_app_id: all_scraped_reviews.extend(scrape_play_store(google_app_id, cutoff_date))
    except Exception as e:
        log(f"⚠️ A critical error occurred during Google Play scraping: {e}. Continuing...")
    try:
        if apple_app_name: all_scraped_reviews.extend(scrape_app_store(apple_app_name, apple_app_country, cutoff_date))
    except Exception as e:
        log(f"⚠️ A critical error occurred during App Store scraping: {e}. Continuing...")

    if not all_scraped_reviews:
        log("No reviews found from any source. Exiting.")
        return

    supabase = get_supabase_client()
    if not supabase: return
    
    try:
        response = supabase.table('raw_reviews').select('id').eq('app_id', primary_app_id).execute()
        existing_ids = {row['id'] for row in response.data}
        log(f"Found {len(existing_ids)} existing reviews in Supabase for '{primary_app_id}'.")
    except Exception as e:
        log(f"❌ Could not fetch existing IDs: {e}")
        return

    reviews_to_process = [r for r in all_scraped_reviews if r['id'] not in existing_ids]
    if not reviews_to_process:
        log("ℹ️ No new reviews to process after checking duplicates.")
        return
    log(f"Found {len(reviews_to_process)} new reviews to analyze and sync.")

    if not GOOGLE_API_KEY:
        log("\n❌ CRITICAL ERROR: GOOGLE_API_KEY not set. Exiting.")
        return
    
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel(GENERATIVE_MODEL)
    except Exception as e:
        log(f"❌ A critical error occurred during LLM setup: {e}")
        return

    total_synced_count = 0
    for i in range(0, len(reviews_to_process), PROCESSING_BATCH_SIZE):
        batch = reviews_to_process[i:i + PROCESSING_BATCH_SIZE]
        log("-" * 50)
        log(f"Processing batch {i//PROCESSING_BATCH_SIZE + 1}...")

        analyzed_batch = analyze_reviews_with_llm(model, batch, topic_list)
        valid_reviews = [r for r in analyzed_batch if r.get('Topic') and r.get('Topic') not in ['N/A', 'Error']]
        
        if not valid_reviews:
            log("No valid reviews in this batch to sync after analysis.")
            continue

        synced_count = sync_raw_reviews_to_supabase(supabase, valid_reviews)
        if synced_count > 0:
            total_synced_count += synced_count
        else:
            log("Batch sync failed. Moving to next batch.")

    log("-" * 50 + f"\n✅ Pipeline finished. Total new reviews synced: {total_synced_count}\n" + "-" * 50)
    st.balloons()


# --- STREAMLIT UI DEFINITION ---
st.set_page_config(page_title="Review Analysis Pipeline", layout="wide")
st.title("📈 App Review Analysis Pipeline")

st.sidebar.header("Pipeline Configuration")
industry_choice = st.sidebar.selectbox("Select Industry", options=list(TOPIC_LISTS.keys()), index=0)
st.sidebar.subheader("App Identifiers")
google_id = st.sidebar.text_input("Google Play App ID", placeholder="e.g., com.google.android.gm")
apple_name = st.sidebar.text_input("Apple App Store Name", placeholder="e.g., gmail-email-by-google")
apple_country_code = st.sidebar.text_input("Apple App Store Country", value="in", help="Use two-letter country codes.")
st.sidebar.write("---")

if st.sidebar.button("🚀 Start Pipeline", type="primary"):
    if not google_id and not apple_name:
        st.sidebar.error("Please provide at least one App ID or Name.")
    else:
        LOGS.clear()  # Clear logs from previous runs
        st.subheader("📋 Pipeline Log")
        LOG_AREA = st.empty() # Create a placeholder for the log output
        run_pipeline(industry_choice, google_id, apple_name, apple_country_code)
else:
    st.info("Configure the pipeline in the sidebar and click 'Start Pipeline' to begin.")