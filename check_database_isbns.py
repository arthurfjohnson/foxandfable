import os
from supabase import create_client, Client
from dotenv import load_dotenv

# --- Configuration & Initialization ---
load_dotenv()
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def check_database_isbns():
    """Check what ISBNs are actually in the database"""
    
    print("Checking what ISBNs are in the database...")
    
    # Search for some common terms to get sample data
    search_terms = ["978", "book", "2024", "2023"]
    
    all_isbns = set()
    
    for term in search_terms:
        print(f"\nSearching for: '{term}'")
        try:
            search_response = supabase.rpc('search_all_books', {'search_query': term}).execute()
            print(f"  Results: {len(search_response.data) if search_response.data else 0}")
            
            if search_response.data:
                for book in search_response.data[:10]:  # Check first 10 results
                    isbn13 = book.get('isbn13')
                    isbn10 = book.get('isbn10')
                    if isbn13:
                        all_isbns.add(isbn13)
                    if isbn10:
                        all_isbns.add(isbn10)
                        
        except Exception as e:
            print(f"  Error: {e}")
    
    print(f"\nFound {len(all_isbns)} unique ISBNs in database")
    print("Sample ISBNs:")
    for i, isbn in enumerate(sorted(list(all_isbns))[:20]):
        print(f"  {i+1}. {isbn}")
    
    # Check if our target ISBN is in the set
    target_isbn = "9780241998137"
    if target_isbn in all_isbns:
        print(f"\n✅ Target ISBN {target_isbn} IS in the database!")
    else:
        print(f"\n❌ Target ISBN {target_isbn} is NOT in the database")
        
        # Look for similar ISBNs
        print("\nLooking for similar ISBNs...")
        similar_isbns = [isbn for isbn in all_isbns if isbn.startswith("978024199")]
        if similar_isbns:
            print(f"Found {len(similar_isbns)} ISBNs starting with 978024199:")
            for isbn in similar_isbns:
                print(f"  - {isbn}")
        else:
            print("No similar ISBNs found")

if __name__ == '__main__':
    check_database_isbns()
