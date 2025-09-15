#!/usr/bin/env python3
"""
Script to split genre columns in all_products_complete_with_categories.xlsx
based on the Genre and Subgenre structure from category_list.csv

This script will transform:
- genre_1, genre_2, genre_3
Into:
- genre_1_1 (Genre), genre_1_2 (Subgenre)
- genre_2_1 (Genre), genre_2_2 (Subgenre)  
- genre_3_1 (Genre), genre_3_2 (Subgenre)
"""

import pandas as pd
import numpy as np
from pathlib import Path

def load_category_mapping():
    """Load the category mapping from category_list.csv"""
    try:
        category_df = pd.read_csv('category_list.csv')
        print(f"Loaded {len(category_df)} categories from category_list.csv")
        
        # Create a mapping dictionary from full category string to (Genre, Subgenre)
        category_map = {}
        for _, row in category_df.iterrows():
            genre = row['Genre']
            subgenre = row['Subgenre']
            
            # Create the full category string as it might appear in the data
            full_category = f"{genre},{subgenre}" if pd.notna(subgenre) else genre
            category_map[full_category] = (genre, subgenre if pd.notna(subgenre) else "")
            
            # Also map just the subgenre in case that's what's stored
            if pd.notna(subgenre):
                category_map[subgenre] = (genre, subgenre)
                
        return category_map
    except Exception as e:
        print(f"Error loading category mapping: {e}")
        return {}

def split_genre_value(genre_value, category_map):
    """
    Split a genre value into (Genre, Subgenre) based on the category mapping
    
    Args:
        genre_value: The genre string to split
        category_map: Dictionary mapping full categories to (Genre, Subgenre)
    
    Returns:
        tuple: (Genre, Subgenre)
    """
    if pd.isna(genre_value) or genre_value == "":
        return ("", "")
    
    genre_str = str(genre_value).strip()
    
    # Try direct lookup first
    if genre_str in category_map:
        return category_map[genre_str]
    
    # Try to find a partial match
    for category, (genre, subgenre) in category_map.items():
        if genre_str in category or category in genre_str:
            return (genre, subgenre)
    
    # If no match found, try to split by common delimiters
    if "," in genre_str:
        parts = genre_str.split(",")
        if len(parts) >= 2:
            return (parts[0].strip(), parts[1].strip())
    
    # If still no match, put everything in the first part
    return (genre_str, "")

def main():
    """Main function to process the Excel file"""
    
    # File paths
    input_file = Path("final_pulls/all_products_complete_with_categories.xlsx")
    output_file = Path("final_pulls/all_products_complete_with_categories_split_genres.xlsx")
    
    print("Starting genre column splitting process...")
    
    # Load category mapping
    print("Loading category mapping...")
    category_map = load_category_mapping()
    if not category_map:
        print("Warning: No category mapping loaded. Proceeding with basic splitting.")
    
    # Load the Excel file
    print(f"Loading Excel file: {input_file}")
    try:
        df = pd.read_excel(input_file)
        print(f"Loaded {len(df)} rows and {len(df.columns)} columns")
        print(f"Columns: {list(df.columns)}")
    except Exception as e:
        print(f"Error loading Excel file: {e}")
        return
    
    # Check if the expected genre columns exist
    genre_columns = ['genre_1', 'genre_2', 'genre_3']
    missing_columns = [col for col in genre_columns if col not in df.columns]
    if missing_columns:
        print(f"Warning: Missing expected columns: {missing_columns}")
        genre_columns = [col for col in genre_columns if col in df.columns]
    
    print(f"Found genre columns: {genre_columns}")
    
    # Process each genre column
    for genre_col in genre_columns:
        print(f"\nProcessing {genre_col}...")
        
        # Create new column names
        genre_part_col = f"{genre_col}_1"
        subgenre_part_col = f"{genre_col}_2"
        
        print(f"Creating columns: {genre_part_col}, {subgenre_part_col}")
        
        # Split the genre values
        genre_parts = []
        subgenre_parts = []
        
        for idx, value in enumerate(df[genre_col]):
            genre_part, subgenre_part = split_genre_value(value, category_map)
            genre_parts.append(genre_part)
            subgenre_parts.append(subgenre_part)
            
            if idx < 5:  # Show first 5 examples
                print(f"  Example {idx+1}: '{value}' -> Genre: '{genre_part}', Subgenre: '{subgenre_part}'")
        
        # Add the new columns to the dataframe
        df[genre_part_col] = genre_parts
        df[subgenre_part_col] = subgenre_parts
        
        print(f"  Created {len(genre_parts)} entries for {genre_part_col}")
        print(f"  Created {len(subgenre_parts)} entries for {subgenre_part_col}")
    
    # Show the new column structure
    print(f"\nNew dataframe has {len(df.columns)} columns:")
    new_genre_cols = [col for col in df.columns if any(col.startswith(f"{base}_") for base in ['genre_1', 'genre_2', 'genre_3'])]
    print("New genre columns:", new_genre_cols)
    
    # Save the updated file
    print(f"\nSaving updated file to: {output_file}")
    try:
        df.to_excel(output_file, index=False)
        print(f"Successfully saved {len(df)} rows to {output_file}")
    except Exception as e:
        print(f"Error saving file: {e}")
        return
    
    # Show some statistics
    print(f"\nProcessing complete!")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Rows processed: {len(df)}")
    
    # Show sample of the results
    print("\nSample of results (first 5 rows):")
    display_cols = ['genre_1', 'genre_1_1', 'genre_1_2', 'genre_2', 'genre_2_1', 'genre_2_2', 'genre_3', 'genre_3_1', 'genre_3_2']
    available_cols = [col for col in display_cols if col in df.columns]
    print(df[available_cols].head().to_string(index=False))

if __name__ == "__main__":
    main()