import pandas as pd
import os
import ast
import re

print("Loading dataset...")
try:
    df = pd.read_parquet('../data/GP_1750_2000_opium_filtered.parquet')
except Exception as e:
    df = pd.read_csv('../data/GP_1750_2000_opium_filtered.csv')

# Extract snippets
books_dir = '../data/opium_books_fulltext'
output_records = []

def parse_keywords(row_kw):
    try:
        kws = ast.literal_eval(row_kw)
        if isinstance(kws, list): return kws
        return [str(kws)]
    except:
        return [x.strip() for x in str(row_kw).split(',')]

for idx, row in df.iterrows():
    book_id = str(row['Etext Number'])
    filepath = os.path.join(books_dir, book_id)
    if not os.path.exists(filepath):
        filepath = os.path.join(books_dir, book_id + '.txt')

    if not os.path.exists(filepath):
        continue

    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
    except Exception as e:
        continue

    # Clean text slightly
    text = re.sub(r'\s+', ' ', text)
    words = text.split()
    
    kws = parse_keywords(row['Opium Keywords'])
    
    # Extract ALL occurrences
    for kw in kws:
        kw_lower = kw.lower()
        for i, word in enumerate(words):
            # Using basic inclusion for matching
            # Filter non-alphanumeric to avoid matching substrings like "shopium"? opium is pretty unique.
            # But "he" or "she" isn't the target here. 
            clean_word = re.sub(r'[^a-z]', '', word.lower())
            if kw_lower == clean_word:
                # Extract 50 left and 50 right
                start_idx = max(0, i - 50)
                end_idx = min(len(words), i + 51)
                
                snippet = " ".join(words[start_idx:end_idx])
                output_records.append({
                    'Book_ID': book_id,
                    'Title': row.get('Title', 'Unknown'),
                    'Authors': row.get('Authors', 'Unknown'),
                    'LoCC': row.get('LoCC', ''),
                    'Keyword': kw,
                    'Snippet': snippet
                })

df_snippets = pd.DataFrame(output_records).drop_duplicates()
print(f"Extracted {len(df_snippets)} total snippets across {df_snippets['Book_ID'].nunique()} unique books.")

output_csv = '../data/snippets.csv'
df_snippets.to_csv(output_csv, index=False, encoding='utf-8-sig')
print(f"Saved to {output_csv}")
