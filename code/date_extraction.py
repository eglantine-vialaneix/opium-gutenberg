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
        converter = 3
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
    
    
    
def clean_gutenberg(df, col_to_keep = ['Etext Number', 'Title', 'Authors', 
                                    'LoCC', 'Bookshelves', 'Subjects', 'rights']):
    
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


def build_anchor_automaton(anchor_index):
    """Build a multi-pattern matcher over anchors for fast substring search."""
    automaton = ahocorasick.Automaton()
    for anchor, matching_book_ids in anchor_index.items():
        automaton.add_word(anchor, tuple(matching_book_ids))

    if len(automaton) == 0:
        return None

    automaton.make_automaton()
    return automaton


def extract_books_fast(df, book_ids, path_to_books):
    candidates = load_fingerprints(book_ids, path_to_books)
    if not candidates:
        return

    anchor_index = build_anchor_index(candidates)
    automaton = build_anchor_automaton(anchor_index)
    if automaton is None:
        return

    found_books = set()
    total_books = len(candidates)

    rows_scanned = 0
    total_rows = len(df)
    pbar = tqdm(total=total_rows, desc="Scanning Chronoberg rows", unit="row")
    pbar.set_postfix(matched_books=f"0/{total_books}")

    try:
        for row in df.iter_rows(named=True):
            rows_scanned += 1
            year = row["year"]
            text = row["text"]

            matched_in_row = set()
            for _, matched_book_ids in automaton.iter(text):
                matched_in_row.update(matched_book_ids)

            new_matches = 0
            for bid in matched_in_row:
                if bid not in found_books:
                    found_books.add(bid)
                    new_matches += 1
                    yield year, bid

            pbar.update(1)
            if new_matches:
                pbar.set_postfix(matched_books=f"{len(found_books)}/{total_books}")

            if len(found_books) == total_books:
                break
    finally:
        if rows_scanned and rows_scanned < total_rows:
            pbar.set_postfix(
                matched_books=f"{len(found_books)}/{total_books}",
                stopped_early=f"row {rows_scanned}/{total_rows}"
            )
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