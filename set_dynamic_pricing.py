#!/usr/bin/env python3
"""
Script to set dynamic pricing based on Gardner RRP values.
Processes the final_pulls/all_products_complete_with_categories.xlsx file.
"""

import os
import pandas as pd
import random
from tqdm import tqdm

def calculate_dynamic_price(gardners_rrp):
    """
    Calculate dynamic price based on Gardner RRP using the specified rules:
    
    - ≤ £7.99: price = gardners_rrp
    - Exactly £10.99: always price = £9.99 (gardners_rrp - 1)
    - £7.99 < gardners_rrp < £10.99: 20% chance (rrp-1), 80% chance (rrp)
    - £10.99 < gardners_rrp ≤ £16.99: 40% chance (rrp-2), 20% chance (rrp-1), 40% chance (rrp)
    - > £16.99: 40% chance (rrp-3), 20% chance (rrp-2), 20% chance (rrp-1), 20% chance (rrp)
    """
    try:
        # Handle empty or invalid values
        if pd.isna(gardners_rrp) or gardners_rrp == '' or gardners_rrp == 0:
            return ''
        
        rrp = float(gardners_rrp)
        
        # Rule 1: Under £7.99 - use full RRP
        if rrp <= 7.99:
            return round(rrp, 2)
        
        # Rule 1.5: Exactly £10.99 - always £1 discount
        elif rrp == 10.99:
            return round(rrp - 1, 2)
        
        # Rule 2: £7.99 < RRP < £10.99 (excluding exactly 10.99)
        elif 7.99 < rrp < 10.99:
            rand = random.random()
            if rand < 0.20:  # 20% chance
                return round(max(0, rrp - 1), 2)
            else:  # 80% chance
                return round(rrp, 2)
        
        # Rule 3: £10.99 < RRP ≤ £16.99
        elif 10.99 < rrp <= 16.99:
            rand = random.random()
            if rand < 0.40:  # 40% chance
                return round(max(0, rrp - 2), 2)
            elif rand < 0.60:  # 20% chance (40% + 20% = 60%)
                return round(max(0, rrp - 1), 2)
            else:  # 40% chance (remaining)
                return round(rrp, 2)
        
        # Rule 4: > £16.99
        else:
            rand = random.random()
            if rand < 0.40:  # 40% chance
                return round(max(0, rrp - 3), 2)
            elif rand < 0.60:  # 20% chance (40% + 20% = 60%)
                return round(max(0, rrp - 2), 2)
            elif rand < 0.80:  # 20% chance (40% + 20% + 20% = 80%)
                return round(max(0, rrp - 1), 2)
            else:  # 20% chance (remaining)
                return round(rrp, 2)
                
    except (ValueError, TypeError):
        return ''

def process_dynamic_pricing():
    """
    Process the Excel file and add dynamic pricing column.
    """
    input_file = 'final_pulls/all_products_complete_with_categories.xlsx'
    
    print(f"📚 Loading {input_file}...")
    
    # Read the Excel file
    df = pd.read_excel(input_file, dtype={'gardners_rrp': float}).fillna('')
    total_rows = len(df)
    
    print(f"📊 Found {total_rows} rows to process")
    
    # Add price column if it doesn't exist
    if 'price' not in df.columns:
        df['price'] = ''
    
    # Track statistics
    processed_count = 0
    pricing_stats = {
        'under_7_99': 0,
        'exactly_10_99': 0,
        'range_7_99_to_10_99': {'full_price': 0, 'minus_1': 0},
        'range_10_99_to_16_99': {'full_price': 0, 'minus_1': 0, 'minus_2': 0},
        'over_16_99': {'full_price': 0, 'minus_1': 0, 'minus_2': 0, 'minus_3': 0},
        'no_rrp': 0
    }
    
    print("💰 Calculating dynamic prices...")
    
    # Set random seed for reproducible results (remove this line for truly random results)
    random.seed(42)
    
    for index, row in tqdm(df.iterrows(), total=total_rows, desc="Setting prices"):
        gardners_rrp = row.get('gardners_rrp', '')
        
        # Skip if no RRP data
        if pd.isna(gardners_rrp) or gardners_rrp == '' or gardners_rrp == 0:
            df.at[index, 'price'] = ''
            pricing_stats['no_rrp'] += 1
            continue
        
        try:
            rrp = float(gardners_rrp)
            new_price = calculate_dynamic_price(rrp)
            df.at[index, 'price'] = new_price
            
            processed_count += 1
            
            # Update statistics
            if rrp <= 7.99:
                pricing_stats['under_7_99'] += 1
            elif rrp == 10.99:
                pricing_stats['exactly_10_99'] += 1
            elif 7.99 < rrp < 10.99:
                if new_price == rrp:
                    pricing_stats['range_7_99_to_10_99']['full_price'] += 1
                else:
                    pricing_stats['range_7_99_to_10_99']['minus_1'] += 1
            elif 10.99 < rrp <= 16.99:
                if new_price == rrp:
                    pricing_stats['range_10_99_to_16_99']['full_price'] += 1
                elif new_price == rrp - 1:
                    pricing_stats['range_10_99_to_16_99']['minus_1'] += 1
                else:
                    pricing_stats['range_10_99_to_16_99']['minus_2'] += 1
            else:  # > 16.99
                if new_price == rrp:
                    pricing_stats['over_16_99']['full_price'] += 1
                elif new_price == rrp - 1:
                    pricing_stats['over_16_99']['minus_1'] += 1
                elif new_price == rrp - 2:
                    pricing_stats['over_16_99']['minus_2'] += 1
                else:
                    pricing_stats['over_16_99']['minus_3'] += 1
                    
        except (ValueError, TypeError):
            df.at[index, 'price'] = ''
            pricing_stats['no_rrp'] += 1
    
    # Save the updated file
    print(f"💾 Saving updated file...")
    df.to_excel(input_file, index=False)
    
    # Print detailed summary
    print("\n" + "="*60)
    print("💰 DYNAMIC PRICING COMPLETE")
    print("="*60)
    print(f"📄 Total rows: {total_rows}")
    print(f"💰 Prices set: {processed_count}")
    print(f"❌ No RRP data: {pricing_stats['no_rrp']}")
    print(f"💾 File updated: {input_file}")
    
    print(f"\n📊 PRICING BREAKDOWN:")
    print(f"  Under £7.99 (full RRP): {pricing_stats['under_7_99']}")
    print(f"  Exactly £10.99 (always £9.99): {pricing_stats['exactly_10_99']}")
    
    total_7_10 = sum(pricing_stats['range_7_99_to_10_99'].values())
    if total_7_10 > 0:
        print(f"  £7.99-£10.98 range: {total_7_10} books")
        print(f"    Full RRP: {pricing_stats['range_7_99_to_10_99']['full_price']} ({pricing_stats['range_7_99_to_10_99']['full_price']/total_7_10*100:.1f}%)")
        print(f"    RRP-£1:   {pricing_stats['range_7_99_to_10_99']['minus_1']} ({pricing_stats['range_7_99_to_10_99']['minus_1']/total_7_10*100:.1f}%)")
    
    total_10_16 = sum(pricing_stats['range_10_99_to_16_99'].values())
    if total_10_16 > 0:
        print(f"  £11.00-£16.99 range: {total_10_16} books")
        print(f"    Full RRP: {pricing_stats['range_10_99_to_16_99']['full_price']} ({pricing_stats['range_10_99_to_16_99']['full_price']/total_10_16*100:.1f}%)")
        print(f"    RRP-£1:   {pricing_stats['range_10_99_to_16_99']['minus_1']} ({pricing_stats['range_10_99_to_16_99']['minus_1']/total_10_16*100:.1f}%)")
        print(f"    RRP-£2:   {pricing_stats['range_10_99_to_16_99']['minus_2']} ({pricing_stats['range_10_99_to_16_99']['minus_2']/total_10_16*100:.1f}%)")
    
    total_over_16 = sum(pricing_stats['over_16_99'].values())
    if total_over_16 > 0:
        print(f"  Over £16.99 range: {total_over_16} books")
        print(f"    Full RRP: {pricing_stats['over_16_99']['full_price']} ({pricing_stats['over_16_99']['full_price']/total_over_16*100:.1f}%)")
        print(f"    RRP-£1:   {pricing_stats['over_16_99']['minus_1']} ({pricing_stats['over_16_99']['minus_1']/total_over_16*100:.1f}%)")
        print(f"    RRP-£2:   {pricing_stats['over_16_99']['minus_2']} ({pricing_stats['over_16_99']['minus_2']/total_over_16*100:.1f}%)")
        print(f"    RRP-£3:   {pricing_stats['over_16_99']['minus_3']} ({pricing_stats['over_16_99']['minus_3']/total_over_16*100:.1f}%)")
    
    return pricing_stats

if __name__ == '__main__':
    print("🚀 Starting dynamic pricing setup...")
    print("="*60)
    print("📝 Pricing Rules:")
    print("   ≤ £7.99: Full RRP")
    print("   Exactly £10.99: Always £9.99 (RRP-£1)")
    print("   £7.99-£10.98: 80% full RRP, 20% RRP-£1")
    print("   £11.00-£16.99: 40% full RRP, 20% RRP-£1, 40% RRP-£2")
    print("   > £16.99: 20% full RRP, 20% RRP-£1, 20% RRP-£2, 40% RRP-£3")
    print("="*60)
    
    try:
        result = process_dynamic_pricing()
        print(f"\n🎉 Success! Dynamic pricing has been applied to all products.")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Please check your file path and data.")
