#!/usr/bin/env python3
"""
OpenAI API script to compare book titles and authors from Excel file.
Processes data in batches for efficiency and includes progress tracking.
"""

import pandas as pd
from openai import OpenAI
import os
from tqdm import tqdm
import time
from typing import List, Dict, Any
import json

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

class BookComparisonAPI:
    def __init__(self, api_key: str = None, batch_size: int = 10):
        """
        Initialize the BookComparisonAPI class.
        
        Args:
            api_key: OpenAI API key (if not provided, will use OPENAI_API_KEY env var)
            batch_size: Number of records to process in each batch
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable or pass api_key parameter.")

        self.client = OpenAI(api_key=self.api_key)
        self.batch_size = batch_size

    def create_comparison_prompt(self, title: str, author: str, bundle_title: str, matched_title: str, matched_author: str) -> str:
        """
        Create a prompt for comparing book information.
        
        Args:
            title: Original title
            author: Original author
            bundle_title: Bundle title
            matched_title: Matched title
            matched_author: Matched author
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""
You are a book matching expert. Compare the following book information and determine if they refer to the same book.

Original Book:
- Title: {title}
- Author: {author}
- Bundle Title: {bundle_title}

Matched Book:
- Title: {matched_title}
- Author: {matched_author}

Consider variations in:
- Title formatting (punctuation, capitalization, subtitles)
- Author name formatting (initials vs full names, order)
- Bundle titles that might contain multiple books
- Minor spelling differences

Respond with ONLY "yes" or "no" - no explanation needed.
"""
        return prompt.strip()

    def compare_books_batch(self, batch_data: List[Dict[str, Any]]) -> List[str]:
        """
        Compare a batch of books using OpenAI API.
        
        Args:
            batch_data: List of dictionaries containing book information
            
        Returns:
            List of "yes" or "no" responses
        """
        responses = []
        
        for row in batch_data:
            try:
                # Create prompt for this specific book
                prompt = self.create_comparison_prompt(
                    title=row.get('title', ''),
                    author=row.get('author', ''),
                    bundle_title=row.get('bundle_title', ''),
                    matched_title=row.get('matched_title', ''),
                    matched_author=row.get('matched_author', '')
                )
                
                # Make individual API call for each book
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1,  # We only need "yes" or "no"
                    temperature=0,  # Deterministic responses
                    timeout=30
                )
                
                # Extract response
                content = response.choices[0].message.content.strip().lower()
                if content in ['yes', 'no']:
                    responses.append(content)
                else:
                    responses.append('no')  # Default to 'no' if unclear response
                    
            except Exception as e:
                print(f"Error processing individual book: {e}")
                responses.append('no')  # Default to 'no' if there's an error
        
        return responses

    def process_excel_file(self, file_path: str, output_path: str = None) -> pd.DataFrame:
        """
        Process the Excel file and add comparison results.
        
        Args:
            file_path: Path to the input Excel file
            output_path: Path to save the output file (optional)
            
        Returns:
            DataFrame with comparison results
        """
        print(f"Loading Excel file: {file_path}")
        df = pd.read_excel(file_path)

        print(f"Found {len(df)} rows to process")
        print(f"Columns: {list(df.columns)}")

        # Check if required columns exist
        required_columns = ['title', 'author', 'bundle_title', 'matched_title', 'matched_author']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            print(f"Warning: Missing columns: {missing_columns}")
            print("Available columns:", list(df.columns))
            return df

        # Initialize results column
        df['is_same_book'] = ''

        # Process in batches
        total_batches = (len(df) + self.batch_size - 1) // self.batch_size

        print(f"Processing {len(df)} records in {total_batches} batches of {self.batch_size}")

        with tqdm(total=len(df), desc="Processing books") as pbar:
            for i in range(0, len(df), self.batch_size):
                batch_end = min(i + self.batch_size, len(df))
                batch_df = df.iloc[i:batch_end]

                # Convert batch to list of dictionaries
                batch_data = batch_df[required_columns].to_dict('records')

                # Get comparison results
                results = self.compare_books_batch(batch_data)

                # Update the dataframe with results
                for j, result in enumerate(results):
                    df.loc[i + j, 'is_same_book'] = result

                pbar.update(len(batch_data))

                # Small delay to avoid rate limiting
                time.sleep(0.1)

        # Save results if output path provided
        if output_path:
            df.to_excel(output_path, index=False)
            print(f"Results saved to: {output_path}")

        return df

def main():
    """Main function to run the book comparison script."""
    # Configuration
    input_file = "individual_titles_with_isbns.xlsx"
    output_file = "individual_titles_with_comparison_results.xlsx"
    batch_size = 50  # Adjust based on your API limits

    try:
        # Initialize the comparison API
        print("Initializing Book Comparison API...")
        api = BookComparisonAPI(batch_size=batch_size)

        # Process the Excel file
        print("Starting book comparison process...")
        results_df = api.process_excel_file(input_file, output_file)

        # Print summary
        if 'is_same_book' in results_df.columns:
            yes_count = (results_df['is_same_book'] == 'yes').sum()
            no_count = (results_df['is_same_book'] == 'no').sum()
            print(f"\nComparison Results Summary:")
            print(f"Same books: {yes_count}")
            print(f"Different books: {no_count}")
            print(f"Total processed: {len(results_df)}")

        print("Book comparison completed successfully!")

    except Exception as e:
        print(f"Error: {e}")
        print("Please check your OpenAI API key and file path.")

if __name__ == "__main__":
    main()