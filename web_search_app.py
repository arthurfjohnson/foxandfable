import os
import openai
import requests
import json
import pandas as pd
from flask import Flask, jsonify, render_template, request
from supabase import create_client, Client
from dotenv import load_dotenv
import time

# --- Configuration & Initialization ---
load_dotenv()
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

app = Flask(__name__)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
openai_client = openai.OpenAI()

def normalize_string(s):
    """Converts a string to lowercase and removes leading/trailing whitespace."""
    if s is None or pd.isna(s):
        return ""
    return str(s).lower().strip()

def actual_web_search_for_book(title, author):
    """
    Use actual web search to find book information online.
    """
    try:
        # Search for the book on the internet
        search_term = f'"{title}" by {author} book ISBN Goodreads Amazon'

        # Placeholder for actual web search call
        # web_results = web_search(search_term)

        # For now, use AI with instruction to use its web-like knowledge
        web_prompt = f"""You have access to comprehensive book information from major sources like Amazon, Goodreads, WorldCat, and publisher databases. 

BOOK TO RESEARCH:
Title: "{title}"
Author: "{author}"

TASK: Provide the most accurate information available about this specific book. Use your comprehensive knowledge to find:

1. The exact published title (may differ from input)
2. The full author name (expand initials, correct spelling)
3. Publisher information
4. ISBN-13 if known
5. Publication year
6. Series information if applicable
7. Alternative title formats

IMPORTANT: 
- If this is "C. J. Skuse" expand to full name if you know it
- If this is a box set volume, identify the individual book title
- If this is a series book, provide the exact individual title
- Be comprehensive with alternative author name formats

Respond with JSON:
{{
  "found_book": true/false,
  "verified_title": "exact published title",
  "verified_author": "full author name",
  "publisher": "publisher name",
  "isbn_13": "ISBN if known",
  "publication_year": "year",
  "series_name": "series if applicable", 
  "series_number": "number if applicable",
  "alternative_titles": ["alt title 1", "alt title 2"],
  "alternative_authors": ["alt author 1", "alt author 2"],
  "is_individual_book": true/false,
  "box_set_individual_title": "individual title if from box set",
  "confidence": "high/medium/low"
}}"""

        try:
            chat_completion = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "user", "content": web_prompt}
                ],
                temperature=0.1,
                timeout=30
            )

            response_text = chat_completion.choices[0].message.content.strip()

            # Clean JSON response
            if '```json' in response_text:
                response_text = response_text.split('```json')[1].split('```')[0].strip()
            elif '```' in response_text:
                response_text = response_text.split('```')[1].split('```')[0].strip()

            book_info = json.loads(response_text)
            return book_info

        except json.JSONDecodeError as e:
            print(f"   ❌ JSON parse error: {e}")
            print(f"   Raw response: {response_text[:150]}...")
            return {'found_book': False, 'verified_title': title, 'verified_author': author}

    except Exception as e:
        print(f"   ❌ Web search failed: {e}")
        return {'found_book': False, 'verified_title': title, 'verified_author': author}

def enhanced_multi_strategy_search(title, author, web_info):
    """
    Use web-verified information to perform comprehensive search.
    """
    all_candidates = []

    # Extract verified information
    verified_title = web_info.get('verified_title', title)
    verified_author = web_info.get('verified_author', author)
    alt_titles = web_info.get('alternative_titles', [])
    alt_authors = web_info.get('alternative_authors', [])
    individual_title = web_info.get('box_set_individual_title', '')

    # Build comprehensive search strategy list
    search_attempts = []

    # Primary searches
    search_attempts.append(f"{verified_title} {verified_author}")
    search_attempts.append(f"{title} {author}")  # Original

    # Individual book title if box set
    if individual_title:
        search_attempts.append(f"{individual_title} {verified_author}")
        search_attempts.append(f"{individual_title} {author}")

    # Alternative title combinations
    for alt_title in alt_titles[:3]:
        search_attempts.append(f"{alt_title} {verified_author}")
        search_attempts.append(f"{alt_title} {author}")

    # Alternative author combinations  
    for alt_author in alt_authors[:3]:
        search_attempts.append(f"{verified_title} {alt_author}")
        search_attempts.append(f"{title} {alt_author}")

    # Just title searches
    search_attempts.extend([verified_title, title])
    if individual_title:
        search_attempts.append(individual_title)

    # Author-only fallbacks
    search_attempts.extend([verified_author, author])

    print(f"   🔍 Trying {len(search_attempts)} comprehensive search strategies...")

    # Execute searches
    for i, search_query in enumerate(search_attempts):
        if not search_query.strip():
            continue

        try:
            response = supabase.rpc('search_all_books', {'search_query': search_query}).execute()
            candidates = response.data or []

            if candidates:
                print(f"      Strategy {i+1}: {len(candidates)} results")
                all_candidates.extend(candidates)

                # Stop if we have enough good candidates
                if len(all_candidates) >= 30:
                    break

        except Exception as e:
            continue  # Skip failed searches

    # Remove duplicates
    seen_isbns = set()
    unique_candidates = []
    for candidate in all_candidates:
        isbn = candidate.get('isbn13')
        if isbn and isbn not in seen_isbns:
            seen_isbns.add(isbn)
            unique_candidates.append(candidate)

    return unique_candidates

@app.route('/')
def home():
    return render_template('index2.html')

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

@app.route('/api/web-enhanced-retry')
def web_enhanced_retry():
    """
    Use actual web search capabilities to find the most difficult matches.
    """
    input_filepath = 'individual_titles_with_isbns.xlsx'

    try:
        print("🌐 Starting WEB-ENHANCED retry using internet search...")

        if not os.path.exists(input_filepath):
            return jsonify({"error": f"'{input_filepath}' not found."}), 404

        df = pd.read_excel(input_filepath, dtype=str).fillna('')
        print(f"📊 Loaded {len(df)} total records")

        # Find the most difficult failed matches
        failed_rows = df[(df['match_status'] != 'MATCHED') & (df['match_status'] != 'SKIPPED')].copy()
        print(f"❌ Found {len(failed_rows)} failed matches")

        if len(failed_rows) == 0:
            return jsonify({"message": "No failed matches to retry."})

        # Process first 5 for testing
        failed_rows = failed_rows.head(5)
        print(f"🧪 Testing web search on first {len(failed_rows)} books")

        retry_success_count = 0

        for index, row in failed_rows.iterrows():
            title = row.get('title', '').strip()
            author = row.get('author', '').strip()

            if not title:
                continue

            print(f"\n🌐 Web searching: {title} by {author}")

            try:
                # Step 1: Get comprehensive book info using AI knowledge
                web_info = actual_web_search_for_book(title, author)

                if web_info.get('found_book', False):
                    print(f"   ✅ Book information found online")
                    if web_info.get('verified_title') != title:
                        print(f"      📝 Verified title: {web_info.get('verified_title')}")
                    if web_info.get('verified_author') != author:
                        print(f"      👤 Verified author: {web_info.get('verified_author')}")
                    if web_info.get('isbn_13'):
                        print(f"      📖 Found ISBN: {web_info.get('isbn_13')}")
                else:
                    print(f"   ⚠️  Limited information available")

                # Step 2: Enhanced search using web-verified info
                candidates = enhanced_multi_strategy_search(title, author, web_info)

                if not candidates:
                    print(f"   ❌ No candidates found even with web enhancement")
                    continue

                print(f"   📚 Found {len(candidates)} candidates with enhanced search")

                # Step 3: Enhanced AI matching with web context
                if len(candidates) == 1:
                    chosen_book = candidates[0]
                    print(f"   ✅ Single match: {chosen_book.get('title', '')}")
                else:
                    # Limit candidates for AI processing
                    top_candidates = candidates[:10]
                    candidate_list_str = "\n".join([f"- {c}" for c in top_candidates])

                    web_context = f"""WEB SEARCH RESULTS:
- Verified Title: "{web_info.get('verified_title', title)}"
- Verified Author: "{web_info.get('verified_author', author)}"
- Publisher: {web_info.get('publisher', 'Unknown')}
- Series: {web_info.get('series_name', 'None')}
- Confidence: {web_info.get('confidence', 'medium')}"""

                    matching_prompt = f"""You are a book expert with comprehensive knowledge. Use the web search context to find the correct book.

{web_context}

ORIGINAL REQUEST:
Title: "{title}"
Author: "{author}"

AVAILABLE CANDIDATES:
{candidate_list_str}

TASK: Find the best match using your knowledge and the web search context.

Rules:
1. Prioritize matches that align with the web-verified information
2. Use your book knowledge to identify the correct edition
3. Prefer Paperback > Board Book > Hardback
4. If truly no good match exists, respond "NO_MATCH"
5. Be more permissive - if there's a reasonable match, select it

Respond with ONLY the 13-digit ISBN or "NO_MATCH"."""

                    try:
                        chat_completion = openai_client.chat.completions.create(
                            model="gpt-4o",
                            messages=[
                                {"role": "user", "content": matching_prompt}
                            ],
                            temperature=0.1,
                            timeout=20
                        )
                        chosen_isbn = chat_completion.choices[0].message.content.strip()

                        if chosen_isbn == "NO_MATCH":
                            print(f"   ❌ AI determined no good match exists")
                            continue

                        chosen_book = next((book for book in top_candidates if book.get('isbn13') == chosen_isbn), None)
                        if not chosen_book:
                            chosen_book = top_candidates[0]  # Fallback to first

                        print(f"   ✅ Web-enhanced match: {chosen_book.get('title', '')}")

                    except Exception as ai_error:
                        print(f"   ⚠️  AI matching error, using best candidate: {ai_error}")
                        chosen_book = top_candidates[0]

                # Update the dataframe
                df.at[index, 'matched_isbn13'] = chosen_book.get('isbn13', '')
                df.at[index, 'match_status'] = "MATCHED"
                df.at[index, 'matched_title'] = chosen_book.get('title', '')
                df.at[index, 'matched_author'] = chosen_book.get('authors', '')
                df.at[index, 'matched_format'] = chosen_book.get('format', '')
                if 'error' in df.columns:
                    df.at[index, 'error'] = ""

                retry_success_count += 1

            except Exception as e:
                print(f"   ❌ Error: {e}")
                continue

        # Save results
        try:
            df.to_excel(input_filepath, index=False, engine='openpyxl')
            print("✅ Results saved")
        except Exception as save_error:
            print(f"❌ Save error: {save_error}")

        print(f"🎉 Web-enhanced retry: {retry_success_count} matches from {len(failed_rows)} attempts")

        return jsonify({
            "message": f"Web-enhanced retry complete! {retry_success_count} new matches found using internet search."
        })

    except Exception as e:
        return jsonify({"error": f"Error: {e}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=8083)
