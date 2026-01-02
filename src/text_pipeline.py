import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

class TextPipeline:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def encode_text(self, text):
        """Encodes a single string into an embedding."""
        if pd.isna(text) or len(str(text).strip()) == 0:
            text = ""
        return self.model.encode(str(text), show_progress_bar=False)

    def get_embeddings(self, series, desc="Encoding text"):
        """Encodes a pandas series of text into a numpy array of embeddings."""
        embeddings = []
        for txt in tqdm(series, desc=desc):
            emb = self.encode_text(txt)
            embeddings.append(emb)
        return np.array(embeddings)

    def process_amenities(self, amenities_series):
        """Processes and encodes a series of parsed amenity lists."""
        amenity_embeddings = []
        for row in tqdm(amenities_series, desc="Encoding amenities"):
            if isinstance(row, list):
                sentence = ", ".join(row)
            elif isinstance(row, str) and row.startswith('['):
                # Handle string representation of list if necessary
                import ast
                try:
                    l = ast.literal_eval(row)
                    sentence = ", ".join(l)
                except:
                    sentence = row
            else:
                sentence = str(row) if pd.notna(row) else ""
            
            emb = self.model.encode(sentence, show_progress_bar=False)
            amenity_embeddings.append(emb)
        return np.array(amenity_embeddings)
