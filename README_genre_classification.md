# Genre Classification Script

This script uses OpenAI's GPT-4o-mini to classify books from `genre_rerun.xlsx` into genres based on your `category_list.csv`.

## Setup

1. Make sure you have an OpenAI API key
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set your OpenAI API key as an environment variable:
   ```bash
   export OPENAI_API_KEY='your-api-key-here'
   ```

## Usage

### Test Run (Recommended First)
Test with just 3 books to make sure everything works:

```bash
python test_genre_classifier.py
```

This will create `test_classification_results.xlsx` with the results.

### Full Processing

**Option 1: Standard Mode (10 books per batch)**
```bash
python genre_classifier.py
```
Creates `classified_genres.xlsx` with all results.

**Option 2: Fast Mode (5 books per batch)** ⚡
```bash
python genre_classifier_fast.py
```
Creates `classified_genres_fast.xlsx` with all results. 
- **2x faster** than standard mode
- Uses smaller batches for quicker API responses
- More API calls but much faster overall

## Progress Tracking

The script includes comprehensive progress tracking:
- **Batch Progress**: Shows current batch number and total batches
- **Book Progress**: Shows individual books processed out of total
- **Success Rate**: Shows how many books in each batch got successfully classified
- **Real-time Stats**: Updates showing total books processed and success rates

You'll see dual progress bars like this:
```
Batch 15/72: 420/3573 Books Success: 48/50 books
Books (current batch: 50 books): 89%|████████▉ | 420/473
```

## Output Format

The output Excel file contains these columns:
- `isbn13` - Book identifier
- `genre_1_1` - First genre (main category)
- `genre_1_2` - First genre (subcategory) 
- `genre_2_1` - Second genre (main category)
- `genre_2_2` - Second genre (subcategory)
- `genre_3_1` - Third genre (main category, optional)
- `genre_3_2` - Third genre (subcategory, optional)
- `age` - Age range (if appropriate)

## Cost Estimation

**Standard Mode (10 books per batch):**
- ~3573 books total
- ~358 batches of 10 books each
- Estimated cost: $15-25 USD

**Fast Mode (5 books per batch):**
- ~3573 books total  
- ~715 batches of 5 books each
- Estimated cost: $20-30 USD
- **Worth it for 2x speed improvement!**

## Speed vs Cost Trade-offs

| Mode | Batch Size | Speed | Cost | Best For |
|------|------------|-------|------|----------|
| **Fast** ⚡ | 5 books | ~21 min | $20-30 | Quick results, don't mind extra cost |
| **Standard** | 10 books | ~42 min | $15-25 | Balanced speed and cost |
| **Economy** | 20 books | ~60+ min | $12-20 | Minimize cost, time not critical |

## Customization

You can modify batch size by editing the `batch_size` parameter:
- `genre_classifier.py` (default is 10)
- `genre_classifier_fast.py` (default is 5)

## Troubleshooting

- If you get rate limit errors, increase the delay between batches in `genre_classifier.py`
- If classifications seem off, you can adjust the temperature parameter in the OpenAI call
- Check that your `category_list.csv` is properly formatted
