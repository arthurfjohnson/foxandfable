#!/usr/bin/env python3
"""
Test script for genre classification - processes just a few books for testing
"""

import pandas as pd
import openai
import json
import os
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Import the functions from the main script
from genre_classifier import load_category_list, classify_books_batch

def test_classification():
    """Test the classification with a small sample."""
    
    # Check API key
    if not os.getenv('OPENAI_API_KEY'):
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key: export OPENAI_API_KEY='your-key-here'")
        return
    
    print("Loading test data...")
    df = pd.read_excel('genre_rerun.xlsx')
    
    # Take just the first 3 books for testing
    test_books = []
    print("\nPreparing test books...")
    for i in tqdm(range(3), desc="Loading books"):
        row = df.iloc[i]
        book = {
            'isbn13': str(row['isbn13']),
            'title': str(row['title']),
            'authors': str(row['authors']),
            'subjects': str(row['subjects']),
            'description': str(row['description'])
        }
        test_books.append(book)
        print(f"  Book {i+1}: {book['title']} by {book['authors']}")
    
    print("\nLoading category list...")
    with tqdm(desc="Loading categories") as pbar:
        category_list = load_category_list()
        pbar.update(1)
    print(f"Loaded {len(category_list.split(chr(10)))} categories")
    
    print("\nClassifying test books...")
    with tqdm(desc="Classifying", total=1) as pbar:
        results = classify_books_batch(test_books, category_list)
        pbar.update(1)
    
    print("\nResults:")
    for result in results:
        print(f"ISBN: {result['isbn13']}")
        if result.get('genre_1_1'):
            print(f"  Genre 1: {result.get('genre_1_1', '')} > {result.get('genre_1_2', '')}")
        if result.get('genre_2_1'):
            print(f"  Genre 2: {result.get('genre_2_1', '')} > {result.get('genre_2_2', '')}")  
        if result.get('genre_3_1'):
            print(f"  Genre 3: {result.get('genre_3_1', '')} > {result.get('genre_3_2', '')}")
        if result.get('age'):
            print(f"  Age: {result.get('age', '')}")
        print()
    
    # Save test results
    print("Saving test results...")
    test_df = pd.DataFrame(results)
    required_columns = ['isbn13', 'genre_1_1', 'genre_1_2', 'genre_2_1', 'genre_2_2', 'genre_3_1', 'genre_3_2', 'age']
    for col in required_columns:
        if col not in test_df.columns:
            test_df[col] = ""
    
    test_df = test_df[required_columns]
    
    with tqdm(desc="Writing Excel file") as pbar:
        test_df.to_excel('test_classification_results.xlsx', index=False)
        pbar.update(1)
    
    print("✅ Test results saved to test_classification_results.xlsx")

if __name__ == "__main__":
    test_classification()



