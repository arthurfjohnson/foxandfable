import os
import openai
import requests
import json
import pandas as pd
from flask import Flask, jsonify, render_template, request
from supabase import create_client, Client
from dotenv import load_dotenv

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

def truncate_candidate_list(base_url, title, author, lean_candidates):
    """
    Dynamically builds a list of candidates that will fit within a URL's length limit.
    """
    final_candidates = []
    # Using a safe URL length limit of 4000 characters
    URL_LIMIT = 4000 
    for candidate in lean_candidates:
        final_candidates.append(candidate)
        temp_candidates_str = json.dumps(final_candidates)
        # Construct the full potential URL to check its length
        temp_url = (
            f"{base_url}api/find-best-isbn?title={requests.utils.quote(title)}"
            f"&author={requests.utils.quote(author or '')}"
            f"&candidates={requests.utils.quote(temp_candidates_str)}"
        )
        if len(temp_url) > URL_LIMIT:
            final_candidates.pop()  # Remove the last item that made it too long
            break
    return final_candidates

def get_gardner_data(gardner_isbn):
    """
    Lookup Gardner data by ISBN to get product_availability and price.
    Returns a dict with gardners_availability and gardners_rrp, or empty values if not found.
    """
    try:
        if not gardner_isbn or pd.isna(gardner_isbn):
            return {'gardners_availability': '', 'gardners_rrp': ''}

        # Clean the ISBN
        clean_isbn = str(gardner_isbn).replace("'", "").strip()

        # Query the gold.all_books table
        response = supabase.schema('gold').table('all_books').select('product_availability,price').eq('isbn13', clean_isbn).execute()

        if response.data and len(response.data) > 0:
            book_data = response.data[0]
            return {
                'gardners_availability': book_data.get('product_availability', ''),
                'gardners_rrp': book_data.get('price', '')
            }
        else:
            return {'gardners_availability': '', 'gardners_rrp': ''}

    except Exception as e:
        print(f"Error looking up Gardner data for ISBN {gardner_isbn}: {e}")
        return {'gardners_availability': '', 'gardners_rrp': ''}

# --- Core Routes ---
@app.route('/')
def home():
    return render_template('index.html')

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
    title = request.args.get('title')
    author = request.args.get('author')
    # MODIFIED: Check if a pre-fetched list of candidates is provided
    candidates_json = request.args.get('candidates')

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

        # If a candidate list is passed in the URL, use it. Otherwise, search the DB.
        if candidates_json:
            candidates = json.loads(candidates_json)
        else:
            search_query = f"{title} {author}" if author else title
            response = supabase.rpc('search_all_books', {'search_query': search_query}).execute()
            candidates = response.data

        if not candidates:
            return jsonify({"error": "No candidate books found."}), 404
        if len(candidates) == 1:
            return jsonify(candidates[0])

        candidate_list_str = "\n".join([f"- {c}" for c in candidates])

        system_prompt = """You are a book-matching expert AI. Your task is to find the best ISBN from a list of search results that matches a user's query.
        You must follow these rules strictly:
        1. Prioritize an exact title and author match.
        2. Prefer Paperback, then Hardback editions.
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
        best_book = next((book for book in candidates if book.get('isbn13') == chosen_isbn), None)

        if not best_book:
            best_book = candidates[0]

        # Add Gardner availability and price data if available
        if best_book and best_book.get('isbn13'):
            gardner_data = get_gardner_data(best_book.get('isbn13'))
            best_book.update(gardner_data)

        return jsonify(best_book)
    except Exception as e:
        return jsonify({"error": f"AI processing error: {e}"}), 500

# --- Batch Processing Routes ---
@app.route('/api/process-new')
def process_new_titles():
    input_filepath = 'individual_titles.xlsx'
    output_filepath = 'processed_titles.xlsx'

    try:
        processed_originals = set()
        df_existing = pd.DataFrame()
        if os.path.exists(output_filepath):
            df_existing = pd.read_excel(output_filepath, dtype=str).fillna('')
            for index, row in df_existing.iterrows():
                processed_originals.add((normalize_string(row.get('title')), normalize_string(row.get('author'))))

        if not os.path.exists(input_filepath):
            return jsonify({"error": f"Input file '{input_filepath}' not found."}), 404

        df_input = pd.read_excel(input_filepath, dtype=str).fillna('')
        new_results = []
        for index, row in df_input.iterrows():
            original_title = row.get('title')
            original_author = row.get('author')

            if not original_title or (normalize_string(original_title), normalize_string(original_author)) in processed_originals:
                continue

            api_url = request.host_url + f"api/find-best-isbn?title={requests.utils.quote(original_title)}&author={requests.utils.quote(original_author or '')}"
            response = requests.get(api_url)
            ai_result = response.json() if response.ok else {"error": response.json().get('error', 'Unknown error')}

            result_columns = {
                'chosen_title': ai_result.get('title'),
                'chosen_isbn': ai_result.get('isbn13'),
                'chosen_author': ai_result.get('authors'),
                'chosen_format': ai_result.get('format'),
                'gardners_availability': ai_result.get('gardners_availability', ''),
                'gardners_rrp': ai_result.get('gardners_rrp', ''),
                'ai_error': ai_result.get('error')
            }
            new_results.append({**row.to_dict(), **result_columns})

        if new_results:
            df_new = pd.DataFrame(new_results)
            df_final = pd.concat([df_existing, df_new], ignore_index=True)
            df_final.to_excel(output_filepath, index=False)

        return jsonify({"message": f"Processing complete. {len(new_results)} new titles were processed and saved to '{output_filepath}'."})

    except Exception as e:
        return jsonify({"error": f"An error occurred: {e}"}), 500

@app.route('/api/recover-isbns')
def recover_isbns():
    input_filepath = 'processed_titles.xlsx'
    output_filepath = 'processed_titles_recovered.xlsx'
    recovered_count = 0

    try:
        if not os.path.exists(input_filepath):
            return jsonify({"error": f"'{input_filepath}' not found."}), 404

        df = pd.read_excel(input_filepath, dtype=str).fillna('')
        for index, row in df.iterrows():
            if row.get('chosen_title') and (not row.get('chosen_isbn') or 'E+' in str(row.get('chosen_isbn'))):
                excel_title_norm = normalize_string(row['chosen_title'])

                response = supabase.rpc('search_all_books', {'search_query': row['chosen_title']}).execute()
                candidates = response.data

                best_match = next((book for book in candidates if normalize_string(book.get('title')) == excel_title_norm), None)

                if best_match:
                    df.at[index, 'chosen_isbn'] = best_match.get('isbn13')
                    df.at[index, 'ai_error'] = ''
                    # Add Gardner availability and price data
                    gardner_data = get_gardner_data(best_match.get('isbn13'))
                    df.at[index, 'gardners_availability'] = gardner_data.get('gardners_availability', '')
                    df.at[index, 'gardners_rrp'] = gardner_data.get('gardners_rrp', '')
                    recovered_count += 1
                else:
                    df.at[index, 'ai_error'] = "Recovery failed: No exact match found in DB."

        df.to_excel(output_filepath, index=False)
        return jsonify({"message": f"Recovery complete. {recovered_count} ISBNs recovered. Check '{output_filepath}'."})

    except Exception as e:
        return jsonify({"error": f"An error occurred during recovery: {e}"}), 500

def handle_retry_logic(row):
    original_title = row.get('title')
    original_author = row.get('author')

    if not original_title:
        return {'ai_error': 'Original title is missing, cannot retry.'}

    api_url = f"{request.host_url}api/find-best-isbn?title={requests.utils.quote(original_title)}&author={requests.utils.quote(original_author or '')}"
    response = requests.get(api_url)
    ai_result = response.json() if response.ok else {"error": response.json().get('error', 'Unknown error')}

    return {
        'chosen_title': ai_result.get('title'),
        'chosen_isbn': ai_result.get('isbn13'),
        'chosen_author': ai_result.get('authors'),
        'chosen_format': ai_result.get('format'),
        'gardners_availability': ai_result.get('gardners_availability', ''),
        'gardners_rrp': ai_result.get('gardners_rrp', ''),
        'ai_error': ai_result.get('error')
    }

@app.route('/api/retry-failed')
def retry_failed_titles():
    input_filepath = 'processed_titles.xlsx'
    retried_count = 0
    try:
        if not os.path.exists(input_filepath):
            return jsonify({"error": f"No '{input_filepath}' file found to retry."}), 404

        df = pd.read_excel(input_filepath, dtype=str).fillna('')

        for index, row in df.iterrows():
            if pd.isna(row.get('chosen_isbn')) or not row.get('chosen_isbn'):
                retried_count += 1
                update_data = handle_retry_logic(row)
                for key, value in update_data.items():
                    df.at[index, key] = value

        if retried_count > 0:
            df.to_excel(input_filepath, index=False)

        return jsonify({"message": f"Retry process complete. {retried_count} titles were re-processed."})
    except Exception as e:
        return jsonify({"error": f"An error occurred: {e}"}), 500

@app.route('/api/retry-by-author')
def retry_by_author():
    """
    MODIFIED: Implements the two-step search:
    1. Searches the DB for all books by the author.
    2. Passes that list to the AI to find the best match for the original title.
    """
    input_filepath = 'processed_titles.xlsx'
    retried_count = 0
    try:
        if not os.path.exists(input_filepath):
            return jsonify({"error": f"No '{input_filepath}' file found to retry."}), 404

        df = pd.read_excel(input_filepath, dtype=str).fillna('')

        for index, row in df.iterrows():
            if (pd.isna(row.get('chosen_isbn')) or not row.get('chosen_isbn')) and row.get('author'):
                retried_count += 1
                original_title = row.get('title')
                original_author = row.get('author')

                # Step 1: Search for all books by the author
                author_search_res = supabase.rpc('search_all_books', {'search_query': original_author}).execute()
                author_books = author_search_res.data

                if author_books:
                    # Create a lean list of candidates for the URL
                    lean_candidates = [{'isbn13': b.get('isbn13'), 'title': b.get('title'), 'authors': b.get('authors'), 'format': b.get('format'), 'publishing_date': b.get('publishing_date')} for b in author_books]
                    safe_candidates = truncate_candidate_list(request.host_url, original_title, original_author, lean_candidates)

                    if not safe_candidates:
                        df.at[index, 'ai_error'] = "Author's book list is too long to process."
                    else:
                        # Step 2: Pass the author's books to the AI to find the best title match
                        candidates_str = json.dumps(safe_candidates)
                        api_url = f"{request.host_url}api/find-best-isbn?title={requests.utils.quote(original_title)}&author={requests.utils.quote(original_author)}&candidates={requests.utils.quote(candidates_str)}"
                        response = requests.get(api_url)
                        ai_result = response.json() if response.ok else {"error": response.json().get('error', 'Unknown error')}

                        update_data = {
                            'chosen_title': ai_result.get('title'),
                            'chosen_isbn': ai_result.get('isbn13'),
                            'chosen_author': ai_result.get('authors'),
                            'chosen_format': ai_result.get('format'),
                            'gardners_availability': ai_result.get('gardners_availability', ''),
                            'gardners_rrp': ai_result.get('gardners_rrp', ''),
                            'ai_error': ai_result.get('error')
                        }
                        for key, value in update_data.items():
                            df.at[index, key] = value
                else:
                    df.at[index, 'ai_error'] = f"No books found in DB for author: {original_author}"

        if retried_count > 0:
            df.to_excel(input_filepath, index=False)

        return jsonify({"message": f"Author-retry process complete. {retried_count} titles were re-processed."})
    except Exception as e:
        return jsonify({"error": f"An error occurred: {e}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=8080)