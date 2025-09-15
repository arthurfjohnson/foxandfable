import os
import openai
import requests
import json
import pandas as pd
from flask import Flask, jsonify, render_template, request
from supabase import create_client, Client
from dotenv import load_dotenv
import time
from tqdm import tqdm

# --- Configuration & Initialization ---
load_dotenv()
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

app = Flask(__name__)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
openai_client = openai.OpenAI()

def normalize_string(s):
    if s is None or pd.isna(s):
        return ""
    return str(s).lower().strip()

def fast_search_book(title, author):
    """Fast book search with multiple strategies, minimal output"""
    try:
        # Try multiple search strategies quickly
        search_terms = [
            f"{title} {author}",  # Original
            title,  # Title only - often works best!
            f"{title} author",  # Title + generic author
            author  # Author fallback
        ]

        all_candidates = []
        for search_term in search_terms:
            try:
                response = supabase.rpc('search_all_books', {'search_query': search_term}).execute()
                candidates = response.data or []
                all_candidates.extend(candidates)

                # If title-only search works, prioritize those results
                if search_term == title and candidates:
                    break

                if len(all_candidates) >= 20:
                    break
            except:
                continue

        if not all_candidates:
            return None

        # Remove duplicates
        seen_isbns = set()
        unique_candidates = []
        for candidate in all_candidates:
            isbn = candidate.get('isbn13')
            if isbn and isbn not in seen_isbns:
                seen_isbns.add(isbn)
                unique_candidates.append(candidate)

        # Find best match using exact title matching first
        title_norm = normalize_string(title)

        # Check for exact matches
        for candidate in unique_candidates:
            candidate_title_norm = normalize_string(candidate.get('title', ''))
            if candidate_title_norm == title_norm:
                return candidate  # Exact match!

            # Check for close matches (title contains or is contained) - but verify quality
        for candidate in unique_candidates:
            candidate_title_norm = normalize_string(candidate.get('title', ''))
            # Only accept close matches if they share significant words
            title_words = set(title_norm.split())
            candidate_words = set(candidate_title_norm.split())
            overlap = len(title_words.intersection(candidate_words)) / max(len(title_words), 1)

            if (title_norm in candidate_title_norm or candidate_title_norm in title_norm) and overlap > 0.5:
                return candidate

        # Score remaining candidates
        title_words = set(title_norm.split())
        best_score = 0
        best_candidate = None

        for candidate in unique_candidates[:10]:
            candidate_words = set(normalize_string(candidate.get('title', '')).split())
            score = len(title_words.intersection(candidate_words)) / max(len(title_words), 1)

            # Bonus for author match
            candidate_author = normalize_string(candidate.get('authors', ''))
            if normalize_string(author) in candidate_author:
                score += 0.3

            if score > best_score:
                best_score = score
                best_candidate = candidate

        return best_candidate if best_score > 0.6 else None  # Only return GOOD matches

    except Exception:
        return None

@app.route('/')
def home():
    return render_template('fast_index.html')

@app.route('/search')
def search():
    query = request.args.get('q')
    try:
        if query:
            try:
                query = query.encode('latin-1').decode('utf-8')
            except (UnicodeEncodeError, UnicodeDecodeError):
                pass
        response = supabase.rpc('search_all_books', {'search_query': query}).execute()
        return jsonify(response.data)
    except Exception as e:
        return jsonify({"error": f"Search error: {e}"}), 500

@app.route('/api/fast-retry')
def fast_retry():
    """
    Super fast batch retry with tqdm progress bars and clean output.
    """
    input_filepath = 'individual_titles_with_isbns.xlsx'

    try:
        print("⚡ Starting FAST batch retry...")

        if not os.path.exists(input_filepath):
            return jsonify({"error": f"'{input_filepath}' not found."}), 404

        df = pd.read_excel(input_filepath, dtype=str).fillna('')

        # Find failed matches
        failed_rows = df[(df['match_status'] != 'MATCHED') & (df['match_status'] != 'SKIPPED')].copy()

        if len(failed_rows) == 0:
            return jsonify({"message": "No failed matches to retry."})

        print(f"📊 Found {len(failed_rows)} failed matches to process")

        # Process with progress bar
        matches_found = 0
        processed = 0
        skipped_low_quality = 0

        with tqdm(total=len(failed_rows), desc="🔍 Fast Retry (0.6+ score)", unit="books", 
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:

            for index, row in failed_rows.iterrows():
                title = row.get('title', '').strip()
                author = row.get('author', '').strip()

                if not title:
                    pbar.update(1)
                    continue

                processed += 1

                # Fast search
                chosen_book = fast_search_book(title, author)

                if chosen_book:
                    # Update dataframe
                    df.at[index, 'matched_isbn13'] = chosen_book.get('isbn13', '')
                    df.at[index, 'match_status'] = "MATCHED"
                    df.at[index, 'matched_title'] = chosen_book.get('title', '')
                    df.at[index, 'matched_author'] = chosen_book.get('authors', '')
                    df.at[index, 'matched_format'] = chosen_book.get('format', '')
                    if 'error' in df.columns:
                        df.at[index, 'error'] = ""

                    matches_found += 1
                else:
                    skipped_low_quality += 1

                # Update progress bar
                success_rate = (matches_found / processed * 100) if processed > 0 else 0
                pbar.set_postfix({
                    'Found': matches_found,
                    'Skipped': skipped_low_quality,
                    'Success': f'{success_rate:.1f}%'
                })
                pbar.update(1)

                # Save progress every 100 books
                if processed % 100 == 0:
                    try:
                        df.to_excel(input_filepath, index=False, engine='openpyxl')
                        tqdm.write(f"💾 Progress: {matches_found} good matches, {skipped_low_quality} low-quality skipped ({success_rate:.1f}% success)")
                    except Exception as e:
                        tqdm.write(f"⚠️  Save error: {e}")

        # Final save
        try:
            df.to_excel(input_filepath, index=False, engine='openpyxl')
            print("✅ Final save completed")
        except Exception as e:
            print(f"❌ Save error: {e}")

        final_success_rate = (matches_found / processed * 100) if processed > 0 else 0
        print(f"🎉 COMPLETE: {matches_found} QUALITY matches, {skipped_low_quality} low-quality skipped ({final_success_rate:.1f}% success)")
        print(f"📊 Quality threshold: 0.6+ score (prevents garbage matches)")

        return jsonify({
            "message": f"Fast retry complete! {matches_found} QUALITY matches found from {processed} books ({final_success_rate:.1f}% success). {skipped_low_quality} low-quality matches skipped."
        })

    except Exception as e:
        print(f"💥 Error: {e}")
        return jsonify({"error": f"An error occurred: {e}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=8084)
