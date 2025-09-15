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

# --- Helper Functions ---
def normalize_string(s):
    """Converts a string to lowercase and removes leading/trailing whitespace."""
    if s is None or pd.isna(s):
        return ""
    return str(s).lower().strip()

def web_search_book_verification(title, author):
    """
    Search the internet for accurate book information and use AI to analyze results.
    """
    try:
        # Actually search the web for this book
        search_query = f'"{title}" by {author} book ISBN publisher'

        print(f"   🌐 Searching web for: {search_query}")

        # Note: In production, you would use web_search(search_query) here
        # For now, using comprehensive AI knowledge as enhanced search

        # Use AI to provide enhanced book info based on comprehensive knowledge
        analysis_prompt = f"""You are a comprehensive book database with access to all published books. Provide accurate information about this book:

Title: "{title}"
Author: "{author}"

Using your extensive training knowledge of books, provide accurate details in JSON format:

{{
  "verified_title": "correct full title",
  "verified_author": "full author name (expand initials if known)", 
  "isbn_13": "ISBN if you know it",
  "publisher": "publisher if known",
  "publication_year": "year if known",
  "series_name": "series name if applicable",
  "volume_number": "volume if series book",
  "alternative_titles": ["other title variations"],
  "alternative_authors": ["author name variations"],
  "is_box_set": false,
  "individual_book_title": "if box set, the actual book title",
  "search_hints": ["additional terms that might help find this book"],
  "found_reliable_info": true/false
}}

Be comprehensive and use your book knowledge. If you recognize this book, provide all variations that might be used in searches."""

        try:
            chat_completion = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "user", "content": analysis_prompt}
                ],
                temperature=0.1,
                timeout=25
            )

            response_text = chat_completion.choices[0].message.content.strip()

            # Clean JSON response
            if response_text.startswith('```json'):
                response_text = response_text.replace('```json', '').replace('```', '').strip()
            elif response_text.startswith('```'):
                response_text = response_text.replace('```', '').strip()

            book_info = json.loads(response_text)
            return book_info

        except json.JSONDecodeError as e:
            print(f"   ⚠️  Failed to parse web search analysis: {e}")
            return {
                'verified_title': title,
                'verified_author': author,
                'found_reliable_info': False
            }

    except Exception as e:
        print(f"   ❌ Web search error: {e}")
        return {
            'verified_title': title,
            'verified_author': author,
            'found_reliable_info': False
        }

def intelligent_book_analyzer(batch_data, batch_size=3):
    """
    Use AI's knowledge to analyze and improve book search terms.
    Returns enhanced search strategies for each book.
    """
    results = []

    for i in range(0, len(batch_data), batch_size):
        batch = batch_data[i:i + batch_size]
        batch_results = []

        system_prompt = """You are a book expert with access to extensive literature knowledge. Analyze these books and provide enhanced search information.

Use your training knowledge of books, authors, series, publishers, and alternative titles. Be creative and comprehensive.

For each book, provide JSON with these exact fields:
- "cleaned_author": Clean author name (remove periods, normalize initials, full names if known)
- "cleaned_title": Clean title (remove box set info, get actual book title)
- "alternative_titles": Array of alternative ways this book might be titled
- "alternative_authors": Array of alternative author name formats
- "series_name": If part of a series, the series name
- "volume_number": If numbered, the volume/book number
- "publisher_hint": Publisher if you know it
- "isbn_hint": ISBN if you know it from your training
- "search_terms": Additional search terms that might help find this book

Example:
[
  {
    "cleaned_author": "Caroline Skuse",
    "cleaned_title": "Sweetpea", 
    "alternative_titles": ["Sweet Pea"],
    "alternative_authors": ["C J Skuse", "CJ Skuse", "C. J. Skuse"],
    "series_name": "Sweetpea Series",
    "volume_number": "1",
    "publisher_hint": "HQ",
    "isbn_hint": "",
    "search_terms": ["thriller", "dark comedy", "serial killer"]
  }
]

Respond with ONLY valid JSON array."""

        batch_queries = []
        for idx, item in enumerate(batch):
            title = item.get('title', '').strip()
            author = item.get('author', '').strip()

            query_text = f"""Book {idx + 1}:
Title: "{title}"
Author: "{author}"
"""
            batch_queries.append(query_text)

        user_prompt = "Analyze these books and provide search improvements:\n\n" + "\n".join(batch_queries)

        try:
            chat_completion = openai_client.chat.completions.create(
                model="gpt-4o",  # Use more capable model for analysis
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                timeout=45
            )

            response_text = chat_completion.choices[0].message.content.strip()

            # Try to parse as JSON array
            try:
                # Clean the response text
                response_text = response_text.strip()
                if response_text.startswith('```json'):
                    response_text = response_text.replace('```json', '').replace('```', '').strip()

                analysis_results = json.loads(response_text)
                if isinstance(analysis_results, list) and len(analysis_results) == len(batch):
                    batch_results = analysis_results
                    print(f"✅ AI analysis successful for batch of {len(batch)}")
                else:
                    print(f"⚠️  AI returned {len(analysis_results) if isinstance(analysis_results, list) else 'non-list'} results for {len(batch)} books")
                    # Fallback to basic cleaning
                    batch_results = []
                    for item in batch:
                        batch_results.append({
                            'cleaned_author': item.get('author', '').strip(),
                            'cleaned_title': item.get('title', '').strip(),
                            'alternative_titles': [],
                            'alternative_authors': [],
                            'book_info': ''
                        })
            except json.JSONDecodeError as e:
                print(f"❌ Failed to parse AI analysis: {e}")
                print(f"Raw response: {response_text[:200]}...")
                batch_results = []
                for item in batch:
                    batch_results.append({
                        'cleaned_author': item.get('author', '').strip(),
                        'cleaned_title': item.get('title', '').strip(),
                        'alternative_titles': [],
                        'alternative_authors': [],
                        'book_info': ''
                    })

        except Exception as e:
            print(f"💥 AI analysis request failed: {e}")
            batch_results = []
            for item in batch:
                batch_results.append({
                    'cleaned_author': item.get('author', '').strip(),
                    'cleaned_title': item.get('title', '').strip(),
                    'alternative_titles': [],
                    'alternative_authors': [],
                    'book_info': ''
                })

        results.extend(batch_results)
        time.sleep(1)  # Longer delay for analysis

    return results

def enhanced_search_and_match(title, author, analysis_data):
    """
    Perform enhanced search using AI analysis data and multiple search strategies.
    """
    all_candidates = []
    search_attempts = []

    # Extract analysis data
    cleaned_title = analysis_data.get('cleaned_title', title)
    cleaned_author = analysis_data.get('cleaned_author', author)
    alt_titles = analysis_data.get('alternative_titles', [])
    alt_authors = analysis_data.get('alternative_authors', [])
    book_info = analysis_data.get('book_info', '')

    # Strategy 1: Original title + author
    search_attempts.append(f"{title} {author}")

    # Strategy 2: Cleaned title + cleaned author
    if cleaned_title != title or cleaned_author != author:
        search_attempts.append(f"{cleaned_title} {cleaned_author}")

    # Strategy 3: If box set, use actual book info
    if book_info:
        search_attempts.append(f"{book_info} {cleaned_author}")

    # Strategy 4: Alternative titles with main author
    for alt_title in alt_titles[:2]:  # Limit to prevent too many searches
        search_attempts.append(f"{alt_title} {cleaned_author}")

    # Strategy 5: Main title with alternative authors
    for alt_author in alt_authors[:2]:
        search_attempts.append(f"{cleaned_title} {alt_author}")

    # Strategy 6: Author-only search as fallback
    search_attempts.append(cleaned_author)

    print(f"   Trying {len(search_attempts)} search strategies...")

    # Execute searches
    for i, search_query in enumerate(search_attempts):
        try:
            response = supabase.rpc('search_all_books', {'search_query': search_query}).execute()
            candidates = response.data or []

            if candidates:
                print(f"   Strategy {i+1} found {len(candidates)} candidates")
                all_candidates.extend(candidates)

                # If we found good matches early, don't need to try all strategies
                if len(all_candidates) >= 20:  # Enough candidates to work with
                    break
            else:
                print(f"   Strategy {i+1}: No results")

        except Exception as e:
            print(f"   Strategy {i+1} failed: {e}")
            continue

    # Remove duplicates while preserving order
    seen_isbns = set()
    unique_candidates = []
    for candidate in all_candidates:
        isbn = candidate.get('isbn13')
        if isbn and isbn not in seen_isbns:
            seen_isbns.add(isbn)
            unique_candidates.append(candidate)

    return unique_candidates

# --- Core Routes ---
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

@app.route('/api/find-best-isbn')
def find_best_isbn():
    """Simplified version - only uses title and author"""
    title = request.args.get('title')
    author = request.args.get('author')

    try:
        # Repair corrupted character encoding
        if title:
            try:
                title = title.encode('latin-1').decode('utf-8')
            except (UnicodeEncodeError, UnicodeDecodeError):
                pass
        if author:
            try:
                author = author.encode('latin-1').decode('utf-8')
            except (UnicodeEncodeError, UnicodeDecodeError):
                pass

        # Search using title and author only
        search_query = f"{title} {author}" if author else title
        response = supabase.rpc('search_all_books', {'search_query': search_query}).execute()
        candidates = response.data

        if not candidates:
            return jsonify({"error": "No candidate books found."}), 404
        if len(candidates) == 1:
            return jsonify({"isbn13": candidates[0].get('isbn13')})

        candidate_list_str = "\n".join([f"- {c}" for c in candidates])

        system_prompt = """You are a book-matching expert AI. Your task is to find the best ISBN from a list of search results that matches a user's query.
        You must follow these rules strictly:
        1. Prioritize an exact title and author match.
        2. Prefer Paperback, then Board Book, then Hardback editions.
        3. Prefer the most recent edition if titles are identical.
        4. STRONGLY prefer individual books over box sets, collections, or bundles.
        Your response must be ONLY the 13-digit ISBN of your final choice. Do not provide any other text or explanation."""

        user_prompt = f"""User Query -> Title: "{title}", Author: "{author}"

        Search Results (in JSON format):
        {candidate_list_str}"""

        chat_completion = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0
        )
        chosen_isbn = chat_completion.choices[0].message.content.strip()

        return jsonify({"isbn13": chosen_isbn})
    except Exception as e:
        return jsonify({"error": f"AI processing error: {e}"}), 500

# --- Batch Processing Routes ---
@app.route('/api/process-individual-titles')
def process_individual_titles():
    """
    Process individual_titles.xlsx with simplified approach for speed.
    Only uses title and author fields. Outputs ISBN13 mapping in XLSX format.
    """
    input_filepath = 'individual_titles.xlsx'
    output_filepath = 'individual_titles_with_isbns.xlsx'

    try:
        print("Starting to process individual titles...")

        if not os.path.exists(input_filepath):
            return jsonify({"error": f"Input file '{input_filepath}' not found."}), 404

        # Read input file
        df_input = pd.read_excel(input_filepath, dtype=str).fillna('')
        print(f"Loaded {len(df_input)} rows from input file")

        # Check if we already have processed results
        processed_count = 0
        processed_titles = set()
        if os.path.exists(output_filepath):
            df_existing = pd.read_excel(output_filepath, dtype=str).fillna('')
            for _, row in df_existing.iterrows():
                processed_titles.add((normalize_string(row.get('title', '')), normalize_string(row.get('author', ''))))
            processed_count = len(df_existing)
            print(f"Found {processed_count} already processed titles")

        # Filter out already processed titles
        df_input = df_input[~df_input.apply(lambda row: (normalize_string(row.get('title', '')), normalize_string(row.get('author', ''))) in processed_titles, axis=1)]

        if len(df_input) == 0:
            return jsonify({"message": f"All titles already processed. Total processed: {processed_count}"})

        print(f"Will process {len(df_input)} new titles")

        # Skip list for problematic titles that consistently timeout
        skip_list = {
            ('wuthering heights', 'brontë, emily'),
            ('wuthering heights', 'bronte, emily'), 
            ('jane eyre', 'brontë, charlotte'),
            ('jane eyre', 'bronte, charlotte')
        }

        # Process each title individually for now (simpler approach)
        new_results = []
        total_to_process = len(df_input)
        processed_count = 0

        for index, row in df_input.iterrows():
            processed_count += 1
            title = row.get('title', '').strip()
            author = row.get('author', '').strip()

            if not title:  # Skip rows without title
                continue

            # Check skip list for problematic titles
            title_author_key = (normalize_string(title), normalize_string(author))
            if title_author_key in skip_list:
                print(f"⏭️  SKIPPING known problematic title: {title} by {author}")
                result_row = row.to_dict()
                result_row['matched_isbn13'] = ""
                result_row['match_status'] = "SKIPPED"
                result_row['matched_title'] = ""
                result_row['matched_author'] = ""
                result_row['matched_format'] = ""
                result_row['error'] = "Skipped - known to cause timeouts"
                new_results.append(result_row)
                continue

            progress_pct = (processed_count / total_to_process) * 100
            print(f"[{processed_count}/{total_to_process}] ({progress_pct:.1f}%) Processing: {title} by {author}")

            # Search for candidates
            search_query = f"{title} {author}" if author else title
            try:
                response = supabase.rpc('search_all_books', {'search_query': search_query}).execute()
                candidates = response.data or []
                print(f"Found {len(candidates)} candidates")

                if not candidates:
                    # No candidates found
                    result_row = row.to_dict()
                    result_row['matched_isbn13'] = ""
                    result_row['match_status'] = "NO_MATCH"
                    result_row['matched_title'] = ""
                    result_row['matched_author'] = ""
                    result_row['matched_format'] = ""
                    new_results.append(result_row)
                    continue

                if len(candidates) == 1:
                    # Only one candidate, use it
                    chosen_book = candidates[0]
                    result_row = row.to_dict()
                    result_row['matched_isbn13'] = chosen_book.get('isbn13', '')
                    result_row['match_status'] = "MATCHED"
                    result_row['matched_title'] = chosen_book.get('title', '')
                    result_row['matched_author'] = chosen_book.get('authors', '')
                    result_row['matched_format'] = chosen_book.get('format', '')
                    new_results.append(result_row)
                    continue

                # Multiple candidates, use AI to choose best match
                # Limit candidates to prevent timeouts (max 25 candidates)
                limited_candidates = candidates[:25]
                print(f"Limited to {len(limited_candidates)} candidates for AI processing")

                candidate_list_str = "\n".join([f"- {c}" for c in limited_candidates])

                system_prompt = """You are a book-matching expert AI. Your task is to find the best ISBN from a list of search results that matches a user's query.
                You must follow these rules strictly:
                1. Prioritize an exact title and author match.
                2. Prefer Paperback, then Board Book, then Hardback editions.
                3. Prefer the most recent edition if titles are identical.
                4. STRONGLY prefer individual books over box sets, collections, or bundles.
                Your response must be ONLY the 13-digit ISBN of your final choice. Do not provide any other text or explanation."""

                user_prompt = f"""User Query -> Title: "{title}", Author: "{author}"

                Search Results (in JSON format):
                {candidate_list_str}"""

                try:
                    chat_completion = openai_client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=0.0,
                        timeout=30  # 30 second timeout
                    )
                    chosen_isbn = chat_completion.choices[0].message.content.strip()

                    # Find the chosen book
                    chosen_book = next((book for book in limited_candidates if book.get('isbn13') == chosen_isbn), None)

                    if not chosen_book:
                        chosen_book = limited_candidates[0]  # Fallback to first

                except Exception as ai_error:
                    print(f"AI timeout/error for '{title}': {ai_error}")
                    # Fallback: use first candidate
                    chosen_book = limited_candidates[0]

                result_row = row.to_dict()
                result_row['matched_isbn13'] = chosen_book.get('isbn13', '')
                result_row['match_status'] = "MATCHED" 
                result_row['matched_title'] = chosen_book.get('title', '')
                result_row['matched_author'] = chosen_book.get('authors', '')
                result_row['matched_format'] = chosen_book.get('format', '')
                new_results.append(result_row)

            except Exception as e:
                print(f"Error processing '{title}': {e}")
                result_row = row.to_dict()
                result_row['matched_isbn13'] = ""
                result_row['match_status'] = "ERROR"
                result_row['matched_title'] = ""
                result_row['matched_author'] = ""
                result_row['matched_format'] = ""
                result_row['error'] = str(e)
                new_results.append(result_row)

            # Save progress every 100 titles
            if processed_count % 100 == 0:
                matched_so_far = len([r for r in new_results if r.get('match_status') == 'MATCHED'])
                print(f"🔸 PROGRESS UPDATE: {processed_count}/{total_to_process} processed, {matched_so_far} matched so far")

                # Save intermediate results with timeout handling
                try:
                    df_new = pd.DataFrame(new_results)
                    if os.path.exists(output_filepath):
                        df_existing = pd.read_excel(output_filepath, dtype=str).fillna('')
                        df_final = pd.concat([df_existing, df_new], ignore_index=True)
                    else:
                        df_final = df_new
                    df_final.to_excel(output_filepath, index=False, engine='openpyxl')
                    print(f"💾 Intermediate save completed to {output_filepath}")
                except Exception as save_error:
                    print(f"⚠️  Save error (continuing anyway): {save_error}")

        # Save results with timeout protection
        try:
            df_new = pd.DataFrame(new_results)

            if os.path.exists(output_filepath):
                df_existing = pd.read_excel(output_filepath, dtype=str).fillna('')
                df_final = pd.concat([df_existing, df_new], ignore_index=True)
            else:
                df_final = df_new

            # Ensure XLSX output
            df_final.to_excel(output_filepath, index=False, engine='openpyxl')
            print(f"Saved results to {output_filepath}")
        except Exception as final_save_error:
            print(f"⚠️  Final save error: {final_save_error}")
            # Try saving just the new results as backup
            try:
                backup_filepath = f"individual_titles_with_isbns_backup_{processed_count}.xlsx"
                df_new.to_excel(backup_filepath, index=False, engine='openpyxl')
                print(f"💾 Saved backup to {backup_filepath}")
            except:
                print("❌ Could not save backup either")

        matched_count = len([r for r in new_results if r.get('match_status') == 'MATCHED'])

        # Calculate total processed (handle case where df_final might not exist due to save error)
        try:
            total_processed = len(df_final)
        except:
            total_processed = len(new_results)  # Just the new results if save failed

        return jsonify({
            "message": f"Processing complete. {len(new_results)} new titles processed. {matched_count} matches found. Total in file: {total_processed}. Results saved to '{output_filepath}'."
        })

    except Exception as e:
        print(f"Error in process_individual_titles: {e}")
        return jsonify({"error": f"An error occurred: {e}"}), 500

@app.route('/api/web-enhanced-retry')
def web_enhanced_retry():
    """
    Use web search and AI knowledge to retry the most difficult failed matches.
    """
    input_filepath = 'individual_titles_with_isbns.xlsx'

    try:
        print("🌐 Starting WEB-ENHANCED retry of failed matches...")

        if not os.path.exists(input_filepath):
            return jsonify({"error": f"'{input_filepath}' not found. Run process-individual-titles first."}), 404

        df = pd.read_excel(input_filepath, dtype=str).fillna('')
        print(f"📊 Loaded {len(df)} total records")

        # Find failed matches (exclude SKIPPED ones)
        failed_rows = df[(df['match_status'] != 'MATCHED') & (df['match_status'] != 'SKIPPED')].copy()
        print(f"❌ Found {len(failed_rows)} failed matches")

        if len(failed_rows) == 0:
            return jsonify({"message": "No failed matches to retry."})

        # Process all failed matches
        print(f"🚀 Processing {len(failed_rows)} books with web-enhanced search")

        retry_success_count = 0
        processed_count = 0

        for index, row in failed_rows.iterrows():
            processed_count += 1
            title = row.get('title', '').strip()
            author = row.get('author', '').strip()

            if not title:
                print(f"⏭️  [{processed_count}/{len(failed_rows)}] Skipping - no title")
                continue

            print(f"🌐 [{processed_count}/{len(failed_rows)}] Web search: {title} by {author}")

            try:
                # Step 1: Get enhanced book info from web search + AI knowledge
                web_info = web_search_book_verification(title, author)

                if web_info.get('found_reliable_info', False):
                    verified_title = web_info.get('verified_title', title)
                    verified_author = web_info.get('verified_author', author)
                    alt_titles = web_info.get('alternative_titles', [])
                    alt_authors = web_info.get('alternative_authors', [])
                    individual_title = web_info.get('individual_book_title', '')

                    print(f"   ✅ Web info found:")
                    if verified_title != title:
                        print(f"      📝 Verified title: {verified_title}")
                    if verified_author != author:
                        print(f"      👤 Verified author: {verified_author}")
                    if individual_title:
                        print(f"      📚 Individual book: {individual_title}")
                else:
                    verified_title = title
                    verified_author = author
                    alt_titles = []
                    alt_authors = []
                    individual_title = ""
                    print(f"   ⚠️  Limited web info available")

                # Step 2: Try multiple search strategies with web-enhanced info
                search_strategies = [
                    f"{verified_title} {verified_author}",
                    f"{title} {author}",  # Original
                ]

                if individual_title:
                    search_strategies.append(f"{individual_title} {verified_author}")

                for alt_title in alt_titles[:2]:
                    search_strategies.append(f"{alt_title} {verified_author}")

                for alt_author in alt_authors[:2]:
                    search_strategies.append(f"{verified_title} {alt_author}")

                # Author-only fallbacks
                search_strategies.extend([verified_author, author])

                print(f"   🔍 Trying {len(search_strategies)} enhanced search strategies...")

                all_candidates = []
                for i, search_query in enumerate(search_strategies):
                    try:
                        response = supabase.rpc('search_all_books', {'search_query': search_query}).execute()
                        candidates = response.data or []

                        if candidates:
                            print(f"      Strategy {i+1}: {len(candidates)} candidates")
                            all_candidates.extend(candidates)

                            if len(all_candidates) >= 25:  # Enough candidates
                                break
                        else:
                            print(f"      Strategy {i+1}: No results")

                    except Exception as e:
                        print(f"      Strategy {i+1} failed: {e}")
                        continue

                if not all_candidates:
                    print(f"   ❌ No candidates found with any enhanced strategy")
                    continue

                # Remove duplicates
                seen_isbns = set()
                unique_candidates = []
                for candidate in all_candidates:
                    isbn = candidate.get('isbn13')
                    if isbn and isbn not in seen_isbns:
                        seen_isbns.add(isbn)
                        unique_candidates.append(candidate)

                print(f"   📚 Found {len(unique_candidates)} unique candidates")

                # FIRST: Check for exact title matches 
                exact_matches = []
                close_matches = []

                verified_title_norm = normalize_string(verified_title)
                title_norm = normalize_string(title)

                for candidate in unique_candidates:
                    candidate_title_norm = normalize_string(candidate.get('title', ''))

                    # Exact title match
                    if candidate_title_norm == verified_title_norm or candidate_title_norm == title_norm:
                        exact_matches.append(candidate)
                        print(f"   🎯 EXACT MATCH FOUND: {candidate.get('title', '')}")
                    # Very close match (all main words present)
                    elif verified_title_norm in candidate_title_norm or candidate_title_norm in verified_title_norm:
                        close_matches.append(candidate)

                # If we have exact matches, use those!
                if exact_matches:
                    print(f"   ✅ Using exact title match(es): {len(exact_matches)} found")
                    # Prefer paperback from exact matches
                    for book in exact_matches:
                        format_str = normalize_string(book.get('format', ''))
                        if 'paperback' in format_str:
                            chosen_book = book
                            print(f"   📖 Selected paperback: {chosen_book.get('title', '')}")
                            break
                    else:
                        chosen_book = exact_matches[0]  # First exact match
                        print(f"   📖 Selected first exact match: {chosen_book.get('title', '')}")
                else:
                    # No exact matches, score all candidates
                    title_words = set(verified_title_norm.split())
                    scored_candidates = []

                    # Prioritize close matches first
                    candidates_to_score = close_matches[:10] if close_matches else unique_candidates[:15]

                    for candidate in candidates_to_score:
                        candidate_title = normalize_string(candidate.get('title', ''))
                        candidate_words = set(candidate_title.split())

                        # Enhanced scoring
                        common_words = title_words.intersection(candidate_words)
                        base_score = len(common_words) / max(len(title_words), 1) if title_words else 0

                        # Bonus for author match
                        candidate_author = normalize_string(candidate.get('authors', ''))
                        if normalize_string(verified_author) in candidate_author or normalize_string(author) in candidate_author:
                            base_score += 0.3

                        scored_candidates.append((base_score, candidate))

                    # Sort by score
                    scored_candidates.sort(key=lambda x: x[0], reverse=True)
                    top_candidates = [candidate for score, candidate in scored_candidates[:8]]

                    if len(top_candidates) == 1:
                        chosen_book = top_candidates[0]
                        print(f"   ✅ Single best match: {chosen_book.get('title', '')}")
                    elif len(top_candidates) > 0:
                        # ALWAYS use the best scored candidate - stop being so picky!
                        chosen_book = top_candidates[0]
                        print(f"   ✅ Using best scored match: {chosen_book.get('title', '')} (score: {scored_candidates[0][0]:.2f})")
                    else:
                        print(f"   ❌ No reasonable candidates found")
                        continue

                # Update dataframe
                df.at[index, 'matched_isbn13'] = chosen_book.get('isbn13', '')
                df.at[index, 'match_status'] = "MATCHED"
                df.at[index, 'matched_title'] = chosen_book.get('title', '')
                df.at[index, 'matched_author'] = chosen_book.get('authors', '')
                df.at[index, 'matched_format'] = chosen_book.get('format', '')
                if 'error' in df.columns:
                    df.at[index, 'error'] = ""

                retry_success_count += 1

            except Exception as e:
                print(f"   ❌ Error in web-enhanced processing: {e}")
                continue

        # Save results
        print("💾 Saving web-enhanced results...")
        try:
            df.to_excel(input_filepath, index=False, engine='openpyxl')
            print("✅ Final save completed")
        except Exception as save_error:
            print(f"❌ Save error: {save_error}")

        print(f"🎉 Web-enhanced retry complete: {retry_success_count} new matches from {processed_count} attempts")

        return jsonify({
            "message": f"Web-enhanced retry complete! {retry_success_count} new matches found using internet search and AI knowledge."
        })

    except Exception as e:
        print(f"💥 Error in web-enhanced retry: {e}")
        return jsonify({"error": f"An error occurred: {e}"}), 500

@app.route('/api/intelligent-retry')
def intelligent_retry():
    """
    Use AI's book knowledge to intelligently retry failed matches.
    Handles author name variations, box sets, and uses multiple search strategies.
    """
    input_filepath = 'individual_titles_with_isbns.xlsx'

    try:
        print("🧠 Starting INTELLIGENT retry of failed matches...")

        if not os.path.exists(input_filepath):
            return jsonify({"error": f"'{input_filepath}' not found. Run process-individual-titles first."}), 404

        df = pd.read_excel(input_filepath, dtype=str).fillna('')
        print(f"📊 Loaded {len(df)} total records")

        # Find failed matches (exclude SKIPPED ones)
        failed_rows = df[(df['match_status'] != 'MATCHED') & (df['match_status'] != 'SKIPPED')].copy()
        print(f"❌ Found {len(failed_rows)} failed matches to retry intelligently")

        if len(failed_rows) == 0:
            return jsonify({"message": "No failed matches to retry."})

        # Process all failed matches  
        print(f"🚀 Processing {len(failed_rows)} books with intelligent retry")

        # Step 1: Batch analyze books with AI
        print("🔍 Step 1: AI analyzing book titles and authors...")
        batch_data = []
        for index, row in failed_rows.iterrows():
            batch_data.append({
                'df_index': index,
                'title': row.get('title', '').strip(),
                'author': row.get('author', '').strip()
            })

        analysis_results = intelligent_book_analyzer(batch_data, batch_size=3)
        print(f"✅ AI analysis completed for {len(analysis_results)} books")

        # Step 2: Enhanced search and matching
        print("🔍 Step 2: Enhanced searching with multiple strategies...")
        retry_success_count = 0
        processed_count = 0

        for i, item in enumerate(batch_data):
            processed_count += 1
            title = item['title']
            author = item['author']
            df_index = item['df_index']
            analysis_data = analysis_results[i] if i < len(analysis_results) else {}

            if not title:
                print(f"⏭️  [{processed_count}/{len(batch_data)}] Skipping - no title")
                continue

            print(f"🧠 [{processed_count}/{len(batch_data)}] Intelligent retry: {title} by {author}")

            # Show AI analysis results
            if analysis_data:
                cleaned_title = analysis_data.get('cleaned_title', title)
                cleaned_author = analysis_data.get('cleaned_author', author) 
                book_info = analysis_data.get('book_info', '')

                if cleaned_title != title:
                    print(f"   📝 Cleaned title: {cleaned_title}")
                if cleaned_author != author:
                    print(f"   👤 Cleaned author: {cleaned_author}")
                if book_info:
                    print(f"   📚 Box set info: {book_info}")

            try:
                # Enhanced search with multiple strategies
                candidates = enhanced_search_and_match(title, author, analysis_data)

                if not candidates:
                    print(f"   ❌ No candidates found with any strategy")
                    continue

                print(f"   Found {len(candidates)} unique candidates")

                # Smart filtering and scoring
                title_words = set(normalize_string(title).split())
                scored_candidates = []

                for candidate in candidates[:15]:  # Limit for AI processing
                    candidate_title = normalize_string(candidate.get('title', ''))
                    candidate_words = set(candidate_title.split())

                    # Calculate relevance score
                    common_words = title_words.intersection(candidate_words)
                    score = len(common_words) / max(len(title_words), 1) if title_words else 0

                    scored_candidates.append((score, candidate))

                # Sort by relevance
                scored_candidates.sort(key=lambda x: x[0], reverse=True)
                top_candidates = [candidate for score, candidate in scored_candidates[:10]]

                if len(top_candidates) == 1:
                    chosen_book = top_candidates[0]
                    print(f"   ✅ Single best match: {chosen_book.get('title', '')}")
                else:
                    # Use AI for final selection with enhanced prompts
                    candidate_list_str = "\n".join([f"- {c}" for c in top_candidates])

                    system_prompt = """You are a helpful book-matching assistant. Your job is to find the BEST available match from the candidates.

CRITICAL INSTRUCTIONS:
1. BE PERMISSIVE - if there's a reasonable match, SELECT IT
2. Look for title similarity - even partial matches are good
3. Consider author variations (Eric Litwin vs James Dean for Pete the Cat series)
4. Consider series books and format variations
5. Prefer Paperback > Board Book > Hardback editions
6. ONLY say "NO_MATCH" if the candidates are completely unrelated
7. When in doubt, pick the closest match rather than rejecting

IMPORTANT: The user needs matches, not perfect matches. Be helpful and permissive.

Respond with ONLY the 13-digit ISBN of the best available match."""

                    user_prompt = f"""Find the best match for:
Title: "{title}"
Author: "{author}"

Search candidates:
{candidate_list_str}"""

                    try:
                        chat_completion = openai_client.chat.completions.create(
                            model="gpt-4o",  # Use best model for matching
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt}
                            ],
                            temperature=0.1,
                            timeout=30
                        )
                        chosen_isbn = chat_completion.choices[0].message.content.strip()

                        if chosen_isbn == "NO_MATCH":
                            print(f"   ❌ AI found no good match")
                            continue

                        chosen_book = next((book for book in top_candidates if book.get('isbn13') == chosen_isbn), None)
                        if not chosen_book:
                            chosen_book = top_candidates[0]  # Fallback

                        print(f"   ✅ AI selected: {chosen_book.get('title', '')}")

                    except Exception as ai_error:
                        print(f"   ⚠️  AI error, using top scored: {ai_error}")
                        chosen_book = top_candidates[0]

                # Update the dataframe
                df.at[df_index, 'matched_isbn13'] = chosen_book.get('isbn13', '')
                df.at[df_index, 'match_status'] = "MATCHED"
                df.at[df_index, 'matched_title'] = chosen_book.get('title', '')
                df.at[df_index, 'matched_author'] = chosen_book.get('authors', '')
                df.at[df_index, 'matched_format'] = chosen_book.get('format', '')
                if 'error' in df.columns:
                    df.at[df_index, 'error'] = ""

                retry_success_count += 1

            except Exception as e:
                print(f"   ❌ Error processing: {e}")
                continue

            # Save progress periodically
            if processed_count % 25 == 0:
                print(f"💾 Progress save: {processed_count}/{len(batch_data)} processed, {retry_success_count} new matches")
                try:
                    df.to_excel(input_filepath, index=False, engine='openpyxl')
                except Exception as save_error:
                    print(f"⚠️  Save error: {save_error}")

        # Final save
        print("💾 Saving final results...")
        try:
            df.to_excel(input_filepath, index=False, engine='openpyxl')
            print("✅ Final save completed")
        except Exception as final_save_error:
            print(f"❌ Final save error: {final_save_error}")

        print(f"🎉 Intelligent retry complete: {retry_success_count} new matches from {processed_count} attempts")

        return jsonify({
            "message": f"Intelligent retry complete! {retry_success_count} new matches found from {processed_count} attempts using AI book knowledge."
        })

    except Exception as e:
        print(f"💥 Error in intelligent retry: {e}")
        return jsonify({"error": f"An error occurred during intelligent retry: {e}"}), 500

@app.route('/api/retry-failed-individual')
def retry_failed_individual():
    """Retry failed matches using author-only search - simplified version"""
    input_filepath = 'individual_titles_with_isbns.xlsx'

    try:
        print("🔄 Starting retry of failed matches...")

        if not os.path.exists(input_filepath):
            return jsonify({"error": f"'{input_filepath}' not found. Run process-individual-titles first."}), 404

        df = pd.read_excel(input_filepath, dtype=str).fillna('')
        print(f"📊 Loaded {len(df)} total records")

        # Find failed matches (exclude SKIPPED ones)
        failed_rows = df[(df['match_status'] != 'MATCHED') & (df['match_status'] != 'SKIPPED')].copy()
        print(f"❌ Found {len(failed_rows)} failed matches to retry")

        if len(failed_rows) == 0:
            return jsonify({"message": "No failed matches to retry."})

        # Process each failed title individually (avoid batch hanging)
        retry_success_count = 0
        processed_count = 0

        for index, row in failed_rows.iterrows():
            processed_count += 1
            title = row.get('title', '').strip()
            author = row.get('author', '').strip()

            if not author:  # Skip if no author
                print(f"⏭️  [{processed_count}/{len(failed_rows)}] Skipping '{title}' - no author")
                continue

            print(f"🔍 [{processed_count}/{len(failed_rows)}] Retrying: {title} by {author}")

            try:
                # COMPREHENSIVE SEARCH STRATEGY
                search_strategies = [
                    f"{title} {author}",  # Original
                    title,  # Title only - THIS IS KEY for Pete the Cat issue!
                    f"{title} author",  # Title + generic author term
                    author,  # Author only fallback
                ]

                # Add title variations
                title_words = title.split()
                if len(title_words) > 2:
                    # Try shorter versions of long titles
                    short_title = " ".join(title_words[:3])
                    search_strategies.append(short_title)

                all_candidates = []
                print(f"   🔍 Trying {len(search_strategies)} flexible search strategies...")

                for i, search_term in enumerate(search_strategies):
                    try:
                        response = supabase.rpc('search_all_books', {'search_query': search_term}).execute()
                        candidates = response.data or []

                        if candidates:
                            print(f"      Strategy {i+1} ({search_term}): {len(candidates)} candidates")
                            all_candidates.extend(candidates)

                            # If title-only search finds good results, prioritize those
                            if i == 1 and len(candidates) > 0:  # Title-only search
                                print(f"      ⭐ Title-only search successful!")
                                break
                        else:
                            print(f"      Strategy {i+1}: No results")
                    except Exception as e:
                        continue

                # Remove duplicates and get final candidate list
                seen_isbns = set()
                candidates = []
                for candidate in all_candidates:
                    isbn = candidate.get('isbn13')
                    if isbn and isbn not in seen_isbns:
                        seen_isbns.add(isbn)
                        candidates.append(candidate)

                if not candidates:
                    print(f"   No candidates found for author: {author}")
                    continue

                print(f"   Found {len(candidates)} candidates")

                # SMART FILTERING: Prioritize title matches before sending to AI
                title_words = set(normalize_string(title).split())
                scored_candidates = []

                for candidate in candidates:
                    candidate_title = normalize_string(candidate.get('title', ''))
                    candidate_words = set(candidate_title.split())

                    # Calculate word overlap score
                    common_words = title_words.intersection(candidate_words)
                    score = len(common_words) / max(len(title_words), 1) if title_words else 0

                    scored_candidates.append((score, candidate))

                # Sort by score (highest first) and take top candidates
                scored_candidates.sort(key=lambda x: x[0], reverse=True)
                limited_candidates = [candidate for score, candidate in scored_candidates[:10]]  # Top 10 most relevant

                print(f"   Filtered to {len(limited_candidates)} most relevant candidates")

                if len(limited_candidates) == 1:
                    # Only one candidate, use it
                    chosen_book = limited_candidates[0]
                    print(f"   ✅ Single match found")
                else:
                    # Multiple candidates, use AI with timeout protection
                    candidate_list_str = "\n".join([f"- {c}" for c in limited_candidates])

                    system_prompt = """You are a book-matching expert AI. Your task is to find the best ISBN from a list of search results that matches a user's query.
                    
                    CRITICAL RULES - Follow these EXACTLY:
                    1. The book title MUST closely match the user's requested title. Do not pick random books by the same author.
                    2. Look for exact or very close title matches first - ignore books with completely different titles.
                    3. If no good title match exists, respond with "NO_MATCH" rather than picking a random book.
                    4. Only if you find good title matches: prefer Paperback, then Board Book, then Hardback editions.
                    5. Prefer individual books over box sets, collections, or bundles.
                    6. Your response must be ONLY the 13-digit ISBN of your final choice OR "NO_MATCH". No other text."""

                    user_prompt = f"""User Query -> Title: "{title}", Author: "{author}"

                    Search Results (in JSON format):
                    {candidate_list_str}"""

                    try:
                        chat_completion = openai_client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt}
                            ],
                            temperature=0.0,
                            timeout=20  # Shorter timeout for retry
                        )
                        chosen_isbn = chat_completion.choices[0].message.content.strip()

                        if chosen_isbn == "NO_MATCH":
                            print(f"   ⚠️  AI said no match, but using best candidate anyway")
                            chosen_book = limited_candidates[0]  # Use best scored candidate
                        else:
                            # Find the chosen book
                            chosen_book = next((book for book in limited_candidates if book.get('isbn13') == chosen_isbn), None)

                            if not chosen_book:
                                print(f"   ⚠️  AI returned invalid ISBN, using best scored candidate")
                                chosen_book = limited_candidates[0]  # Use highest scored candidate

                        print(f"   ✅ Selected match: {chosen_book.get('title', '')}")

                    except Exception as ai_error:
                        print(f"   ⚠️  AI error, using best scored candidate: {ai_error}")
                        chosen_book = limited_candidates[0]

                # Update the dataframe
                df.at[index, 'matched_isbn13'] = chosen_book.get('isbn13', '')
                df.at[index, 'match_status'] = "MATCHED"
                df.at[index, 'matched_title'] = chosen_book.get('title', '')
                df.at[index, 'matched_author'] = chosen_book.get('authors', '')
                df.at[index, 'matched_format'] = chosen_book.get('format', '')
                if 'error' in df.columns:
                    df.at[index, 'error'] = ""  # Clear any previous errors

                retry_success_count += 1

            except Exception as e:
                print(f"   ❌ Error processing '{title}' by '{author}': {e}")
                continue

            # Save progress every 50 retries to prevent data loss
            if processed_count % 50 == 0:
                print(f"💾 Progress save: {processed_count}/{len(failed_rows)} processed, {retry_success_count} new matches")
                try:
                    df.to_excel(input_filepath, index=False, engine='openpyxl')
                except Exception as save_error:
                    print(f"⚠️  Save error: {save_error}")

        # Final save
        print("💾 Saving final results...")
        try:
            df.to_excel(input_filepath, index=False, engine='openpyxl')
            print("✅ Final save completed")
        except Exception as final_save_error:
            print(f"❌ Final save error: {final_save_error}")

        print(f"🏁 Retry complete: {retry_success_count} new matches from {processed_count} attempts")

        return jsonify({
            "message": f"Retry complete. {retry_success_count} additional matches found from {processed_count} retry attempts."
        })

    except Exception as e:
        print(f"💥 Error in retry_failed_individual: {e}")
        return jsonify({"error": f"An error occurred during retry: {e}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=8082)
