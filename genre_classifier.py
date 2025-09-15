#!/usr/bin/env python3
"""
Genre Classification Script using OpenAI 4o-mini
Processes books from genre_rerun.xlsx and classifies them using the category list.
"""

import pandas as pd
import openai
import json
import time
from typing import List, Dict, Any
import os
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Initialize OpenAI client (will be set when needed)
client = None

def load_category_list(filename: str = 'category_list.csv') -> str:
    """Load and format the category list for the prompt."""
    df = pd.read_csv(filename)
    
    # Create a formatted string of categories
    categories = []
    for _, row in df.iterrows():
        genre = row['Genre']
        subgenre = row['Subgenre'] 
        age = row['Age'] if pd.notna(row['Age']) else ""
        
        if age:
            categories.append(f"{genre} > {subgenre} (Age: {age})")
        else:
            categories.append(f"{genre} > {subgenre}")
    
    return "\n".join(categories)

def create_classification_prompt(books: List[Dict], category_list: str) -> str:
    """Create the prompt for OpenAI classification."""
    
    books_text = ""
    for i, book in enumerate(books):
        books_text += f"""
Book {i+1}:
ISBN13: {book['isbn13']}
Title: {book['title']}
Authors: {book['authors']}
Subjects: {book['subjects']}
Description: {book['description']}
"""

    prompt = f"""You are a book classification expert. Classify the following books into 2-3 genres from the provided category list, and assign an appropriate age range if applicable.

CATEGORY LIST:
{category_list}

INSTRUCTIONS:
1. For each book, select 2-3 most appropriate genres from the category list above
2. Each genre should be formatted as: "Main Genre > Subgenre" (exactly as shown in the list)
3. Assign an age range if appropriate (use the age ranges from the category list, or leave blank if not applicable)
4. Return your response as a JSON array with one object per book

REQUIRED JSON FORMAT:
[
  {{
    "isbn13": "book_isbn",
    "genre_1_1": "Main Genre 1",
    "genre_1_2": "Subgenre 1", 
    "genre_2_1": "Main Genre 2",
    "genre_2_2": "Subgenre 2",
    "genre_3_1": "Main Genre 3 (optional)",
    "genre_3_2": "Subgenre 3 (optional)",
    "age": "age_range_or_blank"
  }}
]

BOOKS TO CLASSIFY:
{books_text}

Please classify these books and return the JSON response:"""

    return prompt

def classify_books_batch(books: List[Dict], category_list: str) -> List[Dict]:
    """Classify a batch of books using OpenAI."""
    
    global client
    if client is None:
        client = openai.OpenAI()
    
    prompt = create_classification_prompt(books, category_list)
    
    try:
        # Show that we're making an API call (with progress indication if called from main process)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a precise book classification expert. Always return valid JSON arrays."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=4000
        )
        
        # Parse the JSON response
        content = response.choices[0].message.content
        
        # Clean up the response to extract JSON
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        classifications = json.loads(content)
        
        # Ensure all classifications have required fields
        complete_classifications = [ensure_complete_classification(c) for c in classifications]
        
        return complete_classifications
        
    except Exception as e:
        print(f"Error classifying batch: {e}")
        # Return empty classifications for this batch
        return [{"isbn13": book["isbn13"], "genre_1_1": "", "genre_1_2": "", 
                 "genre_2_1": "", "genre_2_2": "", "genre_3_1": "", "genre_3_2": "", "age": ""} 
                for book in books]

def ensure_complete_classification(classification: Dict) -> Dict:
    """Ensure classification has all required fields."""
    required_fields = {
        'genre_1_1': '', 'genre_1_2': '', 'genre_2_1': '', 'genre_2_2': '', 
        'genre_3_1': '', 'genre_3_2': '', 'age': ''
    }
    
    # Fill missing fields
    for field, default_value in required_fields.items():
        if field not in classification:
            classification[field] = default_value
    
    return classification

def process_books(input_file: str = 'genre_rerun.xlsx', output_file: str = 'classified_genres.xlsx', batch_size: int = 10):
    """Main processing function."""
    
    print("Loading input data...")
    df = pd.read_excel(input_file)
    print(f"Loaded {len(df)} books")
    
    print("Loading category list...")
    category_list = load_category_list()
    
    # Prepare output data
    results = []
    
    # Process in batches
    total_batches = (len(df) + batch_size - 1) // batch_size
    print(f"Processing {total_batches} batches of {batch_size} books each...")
    print(f"Estimated time: {total_batches * 4:.0f}-{total_batches * 8:.0f} seconds (~{(total_batches * 6) / 60:.1f} minutes)")
    
    # Create main progress bar for batches
    batch_pbar = tqdm(total=total_batches, desc="Batches", unit="batch", position=0)
    book_pbar = tqdm(total=len(df), desc="Books", unit="book", position=1)
    
    start_time = time.time()
    
    try:
        for batch_idx in range(0, len(df), batch_size):
            batch_df = df.iloc[batch_idx:batch_idx + batch_size]
            current_batch_num = (batch_idx // batch_size) + 1
            
            # Prepare batch data
            books = []
            for _, row in batch_df.iterrows():
                book = {
                    'isbn13': str(row['isbn13']),
                    'title': str(row['title']),
                    'authors': str(row['authors']),
                    'subjects': str(row['subjects']),
                    'description': str(row['description'])
                }
                books.append(book)
            
            # Update progress bar descriptions with current info
            batch_pbar.set_description(f"Batch {current_batch_num}/{total_batches}")
            book_pbar.set_description(f"Books (current batch: {len(books)} books)")
            
            # Classify the batch
            classifications = classify_books_batch(books, category_list)
            
            # Add to results
            results.extend(classifications)
            
            # Update progress bars
            batch_pbar.update(1)
            book_pbar.update(len(books))
            
            # Calculate and show stats in the progress bar
            elapsed_time = time.time() - start_time
            avg_time_per_batch = elapsed_time / current_batch_num if current_batch_num > 0 else 0
            estimated_remaining = avg_time_per_batch * (total_batches - current_batch_num)
            
            batch_pbar.set_postfix({
                'Books': f"{len(results)}/{len(df)}", 
                'Success': f"{sum(1 for r in classifications if r.get('genre_1_1')) }/{len(classifications)}",
                'ETA': f"{estimated_remaining/60:.1f}m"
            })
            
            # Small delay to respect rate limits (reduced since smaller batches)
            time.sleep(0.5)
            
    finally:
        batch_pbar.close()
        book_pbar.close()
    
    print("Creating output file...")
    
    # Create DataFrame with results
    output_df = pd.DataFrame(results)
    
    # Ensure all required columns exist
    required_columns = ['isbn13', 'genre_1_1', 'genre_1_2', 'genre_2_1', 'genre_2_2', 'genre_3_1', 'genre_3_2', 'age']
    for col in required_columns:
        if col not in output_df.columns:
            output_df[col] = ""
    
    # Reorder columns
    output_df = output_df[required_columns]
    
    # Save to Excel
    output_df.to_excel(output_file, index=False)
    print(f"Results saved to {output_file}")
    print(f"Successfully processed {len(results)} books")

if __name__ == "__main__":
    # Check if OpenAI API key is set
    if not os.getenv('OPENAI_API_KEY'):
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key: export OPENAI_API_KEY='your-key-here'")
        exit(1)
    
    process_books()
