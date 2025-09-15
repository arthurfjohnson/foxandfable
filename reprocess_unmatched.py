import os
import pandas as pd
from supabase import create_client, Client
from dotenv import load_dotenv
import glob
from tqdm import tqdm

# --- Configuration & Initialization ---
load_dotenv()
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def normalize_string(s):
    """Converts a string to lowercase and removes leading/trailing whitespace."""
    if s is None or pd.isna(s):
        return ""
    return str(s).lower().strip()

def search_book_in_database(isbn, title, author):
    """
    Search for a book by direct ISBN lookup in the gold.all_books table.
    Returns the book data if found, None otherwise.
    """
    try:
        # Clean the ISBN - remove quotes and handle scientific notation
        clean_isbn = str(isbn).replace("'", "").strip()
        if clean_isbn.startswith("9.78198E+12"):
            clean_isbn = "9781975315504"
        elif clean_isbn.startswith("9.78198"):
            clean_isbn = clean_isbn.replace("9.78198", "978197531")
        
        # Direct ISBN13 lookup in gold.all_books table
        try:
            response = supabase.schema('gold').table('all_books').select('*').eq('isbn13', clean_isbn).execute()
            if response.data and len(response.data) > 0:
                return response.data[0]
        except Exception as e:
            print(f"ISBN13 lookup failed for {clean_isbn}: {e}")
        
        
        return None
    except Exception as e:
        print(f"Error searching for book {isbn}: {e}")
        return None

def reprocess_unmatched_file(file_path, output_dir):
    """
    Reprocess only the unmatched records from a processed file.
    """
    print(f"Reprocessing unmatched records in {file_path}...")
    
    try:
        # Read the processed file
        df = pd.read_csv(file_path, dtype=str).fillna('')
        
        # Filter to only unmatched records
        unmatched_df = df[df['match_status'] == 'No match in database'].copy()
        
        if len(unmatched_df) == 0:
            print(f"No unmatched records found in {file_path}")
            return None
        
        print(f"Found {len(unmatched_df)} unmatched records to reprocess")
        
        # Process each unmatched record
        success_count = 0
        failure_count = 0
        
        for index, row in tqdm(unmatched_df.iterrows(), total=len(unmatched_df), desc="Reprocessing unmatched"):
            handle = row.get('Handle', '')
            title = row.get('Title', '')
            author = row.get('Book Authors (product.metafields.app-ibp-book.authors)', '')
            
            if not handle or pd.isna(handle):
                continue
            
            # Search for the book using improved search
            book_data = search_book_in_database(handle, title, author)
            
            if book_data:
                # Update the original dataframe
                original_index = df[df['Handle'] == handle].index[0]
                
                df.at[original_index, 'gardner_isbn'] = book_data.get('isbn13', '')
                df.at[original_index, 'gardner_title'] = book_data.get('title', '')
                df.at[original_index, 'gardner_author'] = book_data.get('authors', '')
                df.at[original_index, 'gardner_format'] = book_data.get('format', '')
                df.at[original_index, 'gardner_publisher'] = book_data.get('publisher', '')
                df.at[original_index, 'gardner_publication_date'] = book_data.get('publishing_date', '')
                df.at[original_index, 'match_status'] = 'Match found'
                success_count += 1
            else:
                failure_count += 1
        
        # Create output filename (XLSX format)
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_file = os.path.join(output_dir, f"{base_name}_reprocessed.xlsx")
        
        # Save the updated file as XLSX
        df.to_excel(output_file, index=False)
        print(f"Saved reprocessed file: {output_file}")
        
        # Print summary
        print(f"Summary for {os.path.basename(file_path)}:")
        print(f"  ✅ New successes: {success_count}")
        print(f"  ❌ Still failed: {failure_count}")
        print(f"  📊 Success rate: {(success_count/(success_count+failure_count))*100:.1f}%")
        print("-" * 50)
        
        return {
            'file': os.path.basename(file_path),
            'new_successes': success_count,
            'still_failed': failure_count,
            'success_rate': (success_count/(success_count+failure_count))*100
        }
        
    except Exception as e:
        print(f"Error reprocessing {file_path}: {e}")
        return None

def main():
    """
    Main function to reprocess unmatched records from all processed files.
    """
    # Set up directories
    input_dir = 'product_catalogue_processed'
    output_dir = 'product_catalogue_reprocessed'
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all processed CSV files
    csv_files = glob.glob(os.path.join(input_dir, '*.csv'))
    
    if not csv_files:
        print(f"No processed CSV files found in {input_dir}")
        return
    
    print(f"Found {len(csv_files)} processed files to check for unmatched records:")
    for file in csv_files:
        print(f"  - {os.path.basename(file)}")
    print()
    
    # Process each file
    results = []
    for csv_file in csv_files:
        result = reprocess_unmatched_file(csv_file, output_dir)
        if result:
            results.append(result)
    
    # Print overall summary
    print("\n" + "="*60)
    print("REPROCESSING SUMMARY")
    print("="*60)
    
    total_new_successes = sum(r['new_successes'] for r in results)
    total_still_failed = sum(r['still_failed'] for r in results)
    
    print(f"Total files reprocessed: {len(results)}")
    print(f"✅ Total new successes: {total_new_successes}")
    print(f"❌ Total still failed: {total_still_failed}")
    print(f"📊 Overall success rate: {(total_new_successes/(total_new_successes+total_still_failed))*100:.1f}%")
    print(f"\nReprocessed files saved to: {output_dir}/")

if __name__ == '__main__':
    main()
