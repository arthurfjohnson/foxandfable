#!/usr/bin/env python3
"""
FAST Genre Classification Script using OpenAI 4o-mini
Processes books from genre_rerun.xlsx with smaller batches for faster processing.
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

# Import functions from main script
from genre_classifier import (
    load_category_list, 
    create_classification_prompt, 
    ensure_complete_classification
)

# Initialize OpenAI client (will be set when needed)
client = None

def classify_books_batch_fast(books: List[Dict], category_list: str) -> List[Dict]:
    """Classify a batch of books using OpenAI with faster settings."""
    
    global client
    if client is None:
        client = openai.OpenAI()
    
    prompt = create_classification_prompt(books, category_list)
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a precise book classification expert. Always return valid JSON arrays. Be concise."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=2000  # Reduced for faster response
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

def process_books_fast(input_file: str = 'genre_rerun.xlsx', output_file: str = 'classified_genres_fast.xlsx', batch_size: int = 5):
    """Fast processing with smaller batches."""
    
    print("🚀 FAST MODE: Using smaller batches for quicker processing")
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
    print(f"Estimated time: {total_batches * 2:.0f}-{total_batches * 4:.0f} seconds (~{(total_batches * 3) / 60:.1f} minutes)")
    
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
            classifications = classify_books_batch_fast(books, category_list)
            
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
            
            # Minimal delay for fast processing
            time.sleep(0.2)
            
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
    
    total_time = time.time() - start_time
    print(f"🎉 Results saved to {output_file}")
    print(f"✅ Successfully processed {len(results)} books in {total_time/60:.1f} minutes")
    print(f"⚡ Average: {total_time/len(results):.1f} seconds per book")

if __name__ == "__main__":
    # Check if OpenAI API key is set
    if not os.getenv('OPENAI_API_KEY'):
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key: export OPENAI_API_KEY='your-key-here'")
        exit(1)
    
    process_books_fast()
