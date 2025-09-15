import os
import pandas as pd
from supabase import create_client, Client
from dotenv import load_dotenv
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

def search_gardner_data(isbn):
    """
    Search for a book by ISBN13 in the gold.all_books table and get description.
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
        print(f"Error searching for ISBN {isbn}: {e}")
        return None

def process_existing_titles():
    """
    Process the processed_titles.xlsx file and get Gardner descriptions for each ISBN.
    """
    input_file = 'matched_isbns.xlsx'
    output_file = 'matching_books.xlsx'
    
    print(f"Processing matched ISBNs from {input_file}...")
    
    try:
        # Check if input file exists
        if not os.path.exists(input_file):
            print(f"Error: {input_file} not found.")
            return None
        
        # Read the processed titles file
        df = pd.read_excel(input_file, dtype=str).fillna('')
        
        print(f"Found {len(df)} records to process")
        
        # Create output data list
        output_data = []
        success_count = 0
        failure_count = 0
        
        # Process each record
        for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing existing titles"):
            isbn13 = row.get('isbn13', '')
            
            if not isbn13 or pd.isna(isbn13) or isbn13 == '':
                failure_count += 1
                continue
            
            # Search for Gardner data
            gardner_data = search_gardner_data(isbn13)
            
            if gardner_data:
                # Extract the required fields
                output_record = {
                    'gardners_isbn': gardner_data.get('isbn13', ''),
                    'gardners_title': gardner_data.get('title', ''),
                    'gardners_author': gardner_data.get('authors', ''),
                    'gardners_description': gardner_data.get('description', '')
                }
                output_data.append(output_record)
                success_count += 1
            else:
                failure_count += 1
        
        # Create output DataFrame
        if output_data:
            output_df = pd.DataFrame(output_data)
            
            # Remove duplicates based on ISBN
            output_df = output_df.drop_duplicates(subset=['gardners_isbn'])
            
            # Save to Excel
            output_df.to_excel(output_file, index=False)
            print(f"Saved {len(output_df)} unique records to {output_file}")
        else:
            print("No valid records found to save")
            return None
        
        # Print summary
        print(f"\nSummary:")
        print(f"  Total records processed: {len(df)}")
        print(f"  ✅ Successful matches: {success_count}")
        print(f"  ❌ Failed matches: {failure_count}")
        print(f"  📊 Success rate: {(success_count/(success_count+failure_count))*100:.1f}%")
        print(f"  📄 Unique records saved: {len(output_df)}")
        
        return {
            'total_processed': len(df),
            'successful_matches': success_count,
            'failed_matches': failure_count,
            'unique_records_saved': len(output_df),
            'success_rate': (success_count/(success_count+failure_count))*100
        }
        
    except Exception as e:
        print(f"Error processing file: {e}")
        return None

def main():
    """
    Main function to process existing titles and extract Gardner descriptions.
    """
    print("Processing existing titles to extract Gardner descriptions...")
    print("=" * 60)
    
    result = process_existing_titles()
    
    if result:
        print("\n" + "="*60)
        print("PROCESSING COMPLETE")
        print("="*60)
        print(f"Successfully created matching_books.xlsx with Gardner descriptions")
        print(f"Success rate: {result['success_rate']:.1f}%")
    else:
        print("\n" + "="*60)
        print("PROCESSING FAILED")
        print("="*60)

if __name__ == '__main__':
    main()
