#!/usr/bin/env python3
"""
Process BISAC Split file to remove duplicates and create cleaner dataset.

This script:
1. Loads bisac_split.xlsx
2. Removes duplicate rows with same category_0 and category_1, keeping only the first (general) one
3. Merges category_0 and category_1 with ' / ' separator
4. Saves the result back to bisac_split.xlsx
"""

import pandas as pd
from pathlib import Path

def process_bisac_split():
    """Process the BISAC split file to remove duplicates and merge categories."""
    
    input_file = 'bisac_merge/bisac_split.xlsx'
    
    print("📚 Processing BISAC Split file...")
    print(f"📁 Input file: {input_file}")
    
    # Load the data
    try:
        df = pd.read_excel(input_file)
        print(f"✅ Loaded {len(df)} rows from {input_file}")
        print(f"   Columns: {list(df.columns)}")
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return
    
    # Show some statistics before processing
    print(f"\n📊 Before processing:")
    print(f"   Total rows: {len(df)}")
    
    # Find duplicates
    duplicates_count = df.groupby(['category_0', 'category_1']).size()
    duplicate_groups = duplicates_count[duplicates_count > 1]
    total_duplicates = duplicate_groups.sum() - len(duplicate_groups)  # Total extra rows
    
    print(f"   Duplicate groups: {len(duplicate_groups)}")
    print(f"   Total duplicate rows to remove: {total_duplicates}")
    
    # Show some examples of what will be removed
    print(f"\n🔍 Examples of duplicate groups:")
    for (cat0, cat1), count in duplicate_groups.head(5).items():
        print(f"   '{cat0} / {cat1}': {count} rows (will keep 1, remove {count-1})")
    
    # Remove duplicates - keep only the first occurrence of each category_0 + category_1 combination
    print(f"\n🧹 Removing duplicates...")
    df_deduplicated = df.drop_duplicates(subset=['category_0', 'category_1'], keep='first')
    
    removed_count = len(df) - len(df_deduplicated)
    print(f"   Removed {removed_count} duplicate rows")
    print(f"   Remaining rows: {len(df_deduplicated)}")
    
    # Merge category_0 and category_1 with ' / ' separator
    print(f"\n🔗 Merging categories...")
    df_deduplicated['category'] = df_deduplicated['category_0'] + ' / ' + df_deduplicated['category_1']
    
    # Create the final dataset with just code and merged category
    final_df = df_deduplicated[['code', 'category']].copy()
    
    print(f"✅ Created merged categories")
    print(f"   Final columns: {list(final_df.columns)}")
    
    # Show some examples of the final result
    print(f"\n📋 Sample of final result:")
    print(final_df.head(10).to_string(index=False))
    
    # Save the result back to the same file
    try:
        final_df.to_excel(input_file, index=False)
        print(f"\n💾 Saved processed data to {input_file}")
        print(f"   Final dataset: {len(final_df)} rows, {len(final_df.columns)} columns")
        
        # Verify the save
        verification_df = pd.read_excel(input_file)
        if len(verification_df) == len(final_df):
            print("✅ File saved and verified successfully!")
        else:
            print("⚠️ Warning: Verification failed - file may not have saved correctly")
            
    except Exception as e:
        print(f"❌ Error saving file: {e}")
        return
    
    # Show final statistics
    print(f"\n📊 Processing Summary:")
    print(f"   Original rows: {len(df)}")
    print(f"   Duplicate rows removed: {removed_count}")
    print(f"   Final rows: {len(final_df)}")
    print(f"   Reduction: {removed_count/len(df)*100:.1f}%")
    
    print(f"\n🎉 Processing complete! The AI model will now work with a much cleaner, smaller dataset.")

if __name__ == "__main__":
    process_bisac_split()

