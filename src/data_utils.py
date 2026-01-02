import pandas as pd
import numpy as np
import os
import re
import ast

def load_data(csv_path):
    """Loads dataset from CSV."""
    return pd.read_csv(csv_path)

def clean_dataset(df):
    """Basic cleaning: duplicates and missing values."""
    # Remove duplicate listings on ID or URL
    if "id" in df.columns:
        df.drop_duplicates(subset=["id"], inplace=True)
    elif "url" in df.columns:
        df.drop_duplicates(subset=["url"], inplace=True)
    else:
        df.drop_duplicates(inplace=True)

    # Remove empty rows where price or description is missing
    must_have_cols = []
    if "price" in df.columns: must_have_cols.append("price")
    if "description" in df.columns: must_have_cols.append("description")

    df.dropna(subset=must_have_cols, inplace=True)
    return df

def map_images(df, image_folder):
    """Maps property IDs to local image filenames."""
    if not os.path.exists(image_folder):
        df["local_images"] = "[]"
        return df
    
    all_images = os.listdir(image_folder)
    
    def find_images_for_id(property_id):
        pattern = re.compile(rf"A{property_id}\.\d+\.jpg", re.IGNORECASE)
        matches = [img for img in all_images if pattern.match(img)]
        return matches if matches else []

    df["local_images"] = df["id"].apply(find_images_for_id)
    return df

def clean_column_names(df):
    """Standardizes column names."""
    df.columns = (
        df.columns
          .str.strip()
          .str.lower()
          .str.replace(" ", "_")
          .str.replace("-", "_")
    )
    return df

def convert_numeric_cols(df, cols=None):
    """Converts specified columns to numeric, cleaning strings."""
    if cols is None:
        cols = ["num_of_bedrooms", "num_of_bathrooms", "floor_area", "price"]
    
    for col in cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace("[^0-9.]", "", regex=True)
                .replace("", "nan")
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def extract_city(address):
    """Extracts city from address string."""
    if not isinstance(address, str): return None
    parts = address.split(',')
    return parts[-1].strip() if len(parts) > 1 else None

def clean_text_basic(text):
    """Basic text cleaning for NLP."""
    if not isinstance(text, str): return ""
    t = text.lower()
    t = re.sub(r"[^a-z0-9 ]", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()

def improve_text(text):
    """Advanced text cleaning (removing hashtags, tiny words, numbers)."""
    if not isinstance(text, str): return ""
    t = text.lower()
    t = re.sub(r"#\w+", "", t)  # remove hashtags
    t = re.sub(r"\b\w{1,2}\b", "", t)  # remove tiny useless words
    t = re.sub(r"\d+(\,\d+)*(\.\d+)*", "", t)  # remove big numbers & prices
    t = re.sub(r"[\n\r]", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()

def remove_emojis(text):
    """Removes emojis from text."""
    if not isinstance(text, str):
        return text
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002700-\U000027BF"  # Dingbats
        "\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE
    )
    text = emoji_pattern.sub(r'', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
