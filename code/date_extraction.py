import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shutil
import os
import polars as pl
import warnings
#import re
#from pathlib import Path
import pickle
import ahocorasick
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from tqdm import tqdm




def load_gutenberg(your_path='ENTER_YOUR_PATH_HERE', DOWNLOAD=False):
    """Loading the entire Project Gutenberg dataset from kagglehub. 

    Args:
        your_path (str, optional): If you already downloaded the dataset once, then put your path to kagglehub cache here.
        DOWNLOAD (bool, optional): Downloading the entire dataset from scratch? Defaults to False.

    Returns:
        (df_metadata, path): Pandas DataFrame containing the metadata file + string path of the kaggle cached fodler, containing the metadata and all books.
    """
    
    if DOWNLOAD:
        "Started to download the entire dataset from scratch. This might take some time!"
        path = kagglehub.dataset_download("lokeshparab/gutenberg-books-and-metadata-2025")
    else:
        #your_path = '/Users/eglantinevialaneix' #TODO
        path =  your_path + '/.cache/kagglehub/datasets/lokeshparab/gutenberg-books-and-metadata-2025/versions/4/'
    
    df_metadata = pd.read_csv(path + "/gutenberg_metadata.csv")
    print("Successfully loaded the metadata dataframe.")
    print("Path to cached dataset files:", path)
    
    return df_metadata, path
   
    
    
def check_memory_usage(path="~/.cache/kagglehub", unit="GB"):
    
    if unit == "GB": 
        converter = 3
    elif unit == "MB":
        converter = 2
    else:
        warnings.warn("UnitError: The unit given is not valid. Try 'GB' or 'MB'. Used 'GB' by default.")
        
    total, used, free = shutil.disk_usage(os.path.expanduser(path))
    print(f"Total memory: {total / 1024**converter:.2f} {unit}")
    print(f"Memory used: {used / 1024**converter:.2f} {unit} = {100 * used / total:.2f} %")
    print(f"Free memory: {free / 1024**converter:.2f} {unit} = {100 * free / total:.2f} %")
    
  
    
def check_folder_size(path):
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total += os.path.getsize(fp)

    print(f"Folder size: {total / 1024**3:.2f} GB")
    
    
    
def clean_gutenberg(df, col_to_keep = ['Etext Number', 'Title',
                                    'Bookshelves', 'Authors', 
                                    'rights', 'Subjects']):
    
    new_df = df[df["Language"] == "en"]
    new_df = new_df[new_df["Type"] == "Text"]
    print(f"Dropped {df.shape[0] - new_df.shape[0]} books where Language ≠ english or Type ≠ Text.")
    print(f"Kept only {len(col_to_keep)} columns.")
    
    return new_df[col_to_keep]
    
    
    
def load_chronoberg_json(path_to_file):
    df_big = pl.read_ndjson(path_to_file)
    return df_big


def write_polardf_to_txt_byyear(path_to_write, polar_df):
    for row in polar_df.iter_rows():
        with open(f"{path_to_write}{row[0]}.txt", "w") as file:
            file.write(row[1])



# def load_fingerprint(book_id, path_to_books, fp_size=4000):
#     """Load a book fingerprint used for book-id matching in Chronoberg text."""
#     try:
#         with open(f"{path_to_books}{book_id}", 'r') as f:
#             return f.read(fp_size)[2000:fp_size]
#     except FileNotFoundError:
#         return None


# def load_fingerprints_parallel(book_ids, path_to_books, fp_size=2000, max_workers=16):
#     """Load all fingerprints in parallel using threads (I/O bound)."""
#     candidates = {}
#     with ThreadPoolExecutor(max_workers=max_workers) as executor:
#         future_to_bid = {
#             executor.submit(load_fingerprint, bid, path_to_books, fp_size): bid
#             for bid in book_ids
#         }
#         for future in as_completed(future_to_bid):
#             bid = future_to_bid[future]
#             fp = future.result()
#             if fp:
#                 candidates[bid] = fp
#     return candidates


# def build_automaton(candidates):
#     A = ahocorasick.Automaton()
#     for book_id, fingerprint in candidates.items():
#         if fingerprint:  # extra safety
#             A.add_word(fingerprint, book_id)
#     if len(A) == 0:
#         return None  # <-- key change
#     A.make_automaton()
#     return A


# def extract_books_streaming(
#     df,
#     book_ids,
#     path_to_books,
#     fp_size=2000,
#     show_progress=True,
#     progress_desc="Retrieving dates",
#     max_workers=16,
# ):
#     candidates = load_fingerprints_parallel(book_ids, path_to_books, fp_size, max_workers)
#     print(candidates)
#     if not candidates:
#         return None

#     progress_bar = None
#     if show_progress and tqdm is not None:
#         progress_bar = tqdm(total=len(candidates), desc=progress_desc, unit="book")
#     elif show_progress and tqdm is None:
#         warnings.warn("tqdm is not installed; continuing without a progress bar.")

#     if candidates:
#         automaton = build_automaton(candidates)
#     needs_rebuild = False

#     try:
#         for row in df.iter_rows(named=True):
#             year = row['year']
#             big_string = row['text']

#             if needs_rebuild:
#                 if candidates:  # shouldn't build an empty automaton
#                     automaton = build_automaton(candidates)
#                 else:
#                     break  # nothing left to match
#                 needs_rebuild = False

#             matched = set()
#             for _, book_id in automaton.iter(big_string):
#                 matched.add(book_id)

#             for book_id in matched:
#                 yield year, book_id
#                 del candidates[book_id]
#                 if progress_bar is not None:
#                     progress_bar.update(1)

#             if matched:
#                 needs_rebuild = True

#             if not candidates:
#                 break
#     finally:
#         if progress_bar is not None:
#             progress_bar.close()





def load_fingerprints(book_ids, path, fp_start=2000, fp_end=4000, anchor_size=40, step=200):
    candidates = {}

    for bid in tqdm(book_ids, desc="Loading fingerprints", unit="book"):
        file_path = os.path.join(path, str(bid))
        if not os.path.exists(file_path):
            continue

        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read(fp_end)

        if len(text) < fp_end:
            continue

        fingerprint = text[fp_start:fp_end]

        anchors = [
            fingerprint[i:i+anchor_size]
            for i in range(0, len(fingerprint) - anchor_size, step)
        ]

        if anchors:
            candidates[bid] = anchors

    return candidates



def build_anchor_index(candidates):
    index = defaultdict(set)

    for book_id, anchors in candidates.items():
        for anchor in anchors:
            index[anchor].add(book_id)

    return index


def extract_books_fast(df, book_ids, path_to_books):
    candidates = load_fingerprints(book_ids, path_to_books)
    print("loaded candidates")
    if not candidates:
        print("no candidates!!")
        return 

    anchor_index = build_anchor_index(candidates)
    print("built the anchor")

    found_books = set()
    total_books = len(candidates)
    print("total books = ", total_books)

    pbar = tqdm(total=total_books, desc="Matching books", unit="book")
    print("built pbar")
    
    #i = 0
    for row in df.iter_rows(named=True):
        #print(i)
        year = row["year"]
        text = row["text"]
        #j = 0
        for anchor, book_ids in anchor_index.items():
            #print(j)
            if anchor in text:
                for bid in book_ids:
                    if bid not in found_books:
                        found_books.add(bid)
                        pbar.update(1)
                        yield year, bid
            #j += 1

        if len(found_books) == total_books:
            break
        #i += 1

    pbar.close()











def save_dict_to_pickle(dict, name, path_to_write):
    with open(path_to_write + name, "wb") as f:
        pickle.dump(dict, f)
    print("Saved pickle file at:", path_to_write)
        
        
def read_pickle(file_name):
    """Loads a pickle file from the given location.

    Args:
        file_name (str): file name (with path to file before)

    Returns:
        pkl_dict: dictionary of the pickle file that has been loaded
    """
    with open(file_name, "rb") as f:
        pkl_dict = pickle.load(f)
    return pkl_dict