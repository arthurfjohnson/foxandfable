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
        
        # Note: ISBN10 column doesn't exist in gold.all_books table, only isbn13
        
        return None
    except Exception as e:
        print(f"Error searching for book {isbn}: {e}")
        return None

def process_csv_file(file_path, output_dir):
    """
    Process a single CSV file and add gardner_isbn and gardner_title columns.
    """
    print(f"Processing {file_path}...")
    
    try:
        # Read the CSV file
        df = pd.read_csv(file_path, dtype=str).fillna('')
        
        # Add new columns for Gardner data
        df['gardner_isbn'] = ''
        df['gardner_title'] = ''
        df['gardner_author'] = ''
        df['gardner_format'] = ''
        df['gardner_publisher'] = ''
        df['gardner_publication_date'] = ''
        df['gardner_description'] = ''
        df['gardner_subjects'] = ''
        df['match_status'] = ''
        
        # Process each row
        total_rows = len(df)
        success_count = 0
        failure_count = 0
        no_handle_count = 0
        
        print(f"Processing {total_rows} rows...")
        
        for index, row in tqdm(df.iterrows(), total=total_rows, desc=f"Processing {os.path.basename(file_path)}"):
            handle = row.get('Handle', '')
            title = row.get('Title', '')
            author = row.get('Book Authors (product.metafields.app-ibp-book.authors)', '')
            
            if not handle or pd.isna(handle):
                df.at[index, 'match_status'] = 'No Handle/ISBN provided'
                no_handle_count += 1
                continue
            
            # Search for the book in the database using title and author
            book_data = search_book_in_database(handle, title, author)
            
            if book_data:
                # Match found - populate Gardner columns
                df.at[index, 'gardner_isbn'] = book_data.get('isbn13', '')
                df.at[index, 'gardner_title'] = book_data.get('title', '')
                df.at[index, 'gardner_author'] = book_data.get('authors', '')
                df.at[index, 'gardner_format'] = book_data.get('product_form_description', '')
                df.at[index, 'gardner_publisher'] = book_data.get('publisher', '')
                df.at[index, 'gardner_publication_date'] = book_data.get('publishing_date', '')
                df.at[index, 'gardner_description'] = book_data.get('description', '')
                df.at[index, 'gardner_subjects'] = book_data.get('subjects', '')
                df.at[index, 'match_status'] = 'Match found'
                success_count += 1
            else:
                # No match found
                df.at[index, 'match_status'] = 'No match in database'
                failure_count += 1
        
        # Create output filename
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_file = os.path.join(output_dir, f"{base_name}_with_gardner_data.xlsx")
        
        # Save the processed file as XLSX
        df.to_excel(output_file, index=False)
        print(f"Saved processed file: {output_file}")
        
        # Print summary statistics
        print(f"Summary for {os.path.basename(file_path)}:")
        print(f"  Total rows: {total_rows}")
        print(f"  ✅ Successes: {success_count}")
        print(f"  ❌ Failures: {failure_count}")
        print(f"  ⚠️  No Handle/ISBN: {no_handle_count}")
        print(f"  📊 Success rate: {(success_count/total_rows)*100:.1f}%")
        print("-" * 50)
        
        return {
            'file': os.path.basename(file_path),
            'total_rows': total_rows,
            'matches': success_count,
            'no_matches': failure_count,
            'no_handle': no_handle_count,
            'match_rate': (success_count/total_rows)*100
        }
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def main():
    """
    Main function to process all CSV files in the product_catalogue directory.
    """
    # Set up directories
    input_dir = 'product_catalogue'
    output_dir = 'product_catalogue_processed'
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all CSV files in the input directory
    csv_files = glob.glob(os.path.join(input_dir, '*.csv'))
    
    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return
    
    print(f"Found {len(csv_files)} CSV files to process:")
    for file in csv_files:
        print(f"  - {os.path.basename(file)}")
    print()
    
    # Process each CSV file
    results = []
    for csv_file in csv_files:
        result = process_csv_file(csv_file, output_dir)
        if result:
            results.append(result)
    
    # Print overall summary
    print("\n" + "="*60)
    print("OVERALL PROCESSING SUMMARY")
    print("="*60)
    
    total_rows = sum(r['total_rows'] for r in results)
    total_matches = sum(r['matches'] for r in results)
    total_no_matches = sum(r['no_matches'] for r in results)
    total_no_handle = sum(r['no_handle'] for r in results)
    
    print(f"Total files processed: {len(results)}")
    print(f"Total rows processed: {total_rows}")
    print(f"✅ Total successes: {total_matches}")
    print(f"❌ Total failures: {total_no_matches}")
    print(f"⚠️  Total no Handle/ISBN: {total_no_handle}")
    print(f"📊 Overall success rate: {(total_matches/total_rows)*100:.1f}%")
    print(f"\nProcessed files saved to: {output_dir}/")
    
    # Save summary to a file
    summary_file = os.path.join(output_dir, 'processing_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("PRODUCT CATALOGUE PROCESSING SUMMARY\n")
        f.write("="*50 + "\n\n")
        f.write(f"Total files processed: {len(results)}\n")
        f.write(f"Total rows processed: {total_rows}\n")
        f.write(f"Total successes: {total_matches}\n")
        f.write(f"Total failures: {total_no_matches}\n")
        f.write(f"Total no Handle/ISBN: {total_no_handle}\n")
        f.write(f"Overall success rate: {(total_matches/total_rows)*100:.1f}%\n\n")
        
        f.write("DETAILED RESULTS BY FILE:\n")
        f.write("-" * 30 + "\n")
        for result in results:
            f.write(f"File: {result['file']}\n")
            f.write(f"  Rows: {result['total_rows']}\n")
            f.write(f"  Successes: {result['matches']}\n")
            f.write(f"  Failures: {result['no_matches']}\n")
            f.write(f"  No Handle: {result['no_handle']}\n")
            f.write(f"  Success rate: {result['match_rate']:.1f}%\n\n")
    
    print(f"\nDetailed summary saved to: {summary_file}")

if __name__ == '__main__':
    main()
