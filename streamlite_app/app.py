# app_fixed.py — Fixed & more robust Streamlit app for Rental Price Estimator
# Changes made: safer imports, handle None uploaded_images, robust joblib loading,
# safer preprocessor transform handling, default transforms, clearer logs, and
# improved pad/truncate behavior.

import streamlit as st
import os
import joblib
import numpy as np
import pandas as pd
from PIL import Image
import traceback

st.set_page_config(page_title="Rental Price Estimator", layout="centered")

st.title("Rental Price Estimator")
st.write(
    "Multimodal app (text + images + tabular). The app will attempt to load your saved model and preprocessor. "
    "If files are missing, the app will use safe fallbacks and explain what's happening."
)

# -----------------------------
# Utilities
# -----------------------------

def safe_load_joblib(path):
    try:
        return joblib.load(path)
    except Exception:
        return None


def get_model_expected_dim(model):
    try:
        if hasattr(model, "n_features_in_"):
            return int(model.n_features_in_)
        # XGBoost Booster (sklearn API)
        if hasattr(model, "get_booster"):
            try:
                return int(model.get_booster().num_features())
            except Exception:
                pass
        if hasattr(model, "booster_"):
            try:
                return int(model.booster_.num_features())
            except Exception:
                pass
    except Exception:
        pass
    return None


def pad_or_truncate(vec, expected):
    v = np.asarray(vec).ravel()
    cur = v.shape[0]
    if cur == expected:
        return v
    if cur < expected:
        pad = np.zeros(expected - cur, dtype=v.dtype)
        return np.concatenate([v, pad], axis=0)
    else:
        return v[:expected]


def ensure_2d(a):
    a = np.asarray(a)
    if a.ndim == 1:
        return a.reshape(1, -1)
    return a

# -----------------------------
# Try to import embedding libs (optional)
# -----------------------------
HAS_SENTENCE = False
HAS_TORCH = False
HAS_TORCHVISION = False
clip_encoder = None
text_encoder = None
torch = None
torchvision = None
efficientnet_model = None
img_transform = None

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE = True
except Exception:
    HAS_SENTENCE = False

# Try torch/torchvision for EfficientNet
try:
    import torch
    import torchvision
    from torchvision import transforms
    HAS_TORCH = True
    HAS_TORCHVISION = True
except Exception:
    HAS_TORCH = False
    HAS_TORCHVISION = False

if HAS_TORCH and HAS_TORCHVISION:
    try:
        from torchvision import models as tv_models
        # Newer torchvision API provides weights enums; handle gracefully
        try:
            w = tv_models.EfficientNet_B0_Weights.IMAGENET1K_V1
            efficientnet_model = tv_models.efficientnet_b0(weights=w)
        except Exception:
            # fallback to old call
            efficientnet_model = tv_models.efficientnet_b0(pretrained=True)
        # remove classifier to extract features
        if hasattr(efficientnet_model, "classifier"):
            efficientnet_model.classifier = torch.nn.Identity()
        elif hasattr(efficientnet_model, "fc"):
            efficientnet_model.fc = torch.nn.Identity()
        efficientnet_model.eval()
        img_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    except Exception:
        efficientnet_model = None
        img_transform = None

# Load sentence-transformers if present
if HAS_SENTENCE:
    try:
        text_encoder = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        text_encoder = None

# Fallback to CLIP (sentence-transformers implementation)
if HAS_SENTENCE and text_encoder is None:
    try:
        clip_encoder = SentenceTransformer("clip-ViT-B-32")
    except Exception:
        clip_encoder = None

# -----------------------------
# Load model(s) & preprocessor if present
# -----------------------------
MODEL_DIR = "models"
MODEL_PATHS = []
if os.path.exists(MODEL_DIR):
    MODEL_PATHS = [p for p in os.listdir(MODEL_DIR) if p.startswith("best_model_") and p.endswith(".joblib")]

MODEL_PATH = None
if MODEL_PATHS:
    MODEL_PATH = os.path.join(MODEL_DIR, MODEL_PATHS[0])

preprocessor_path = os.path.join(MODEL_DIR, "preprocessing_pipeline.joblib")

loaded_model = None
preprocessor = None

if MODEL_PATH and os.path.exists(MODEL_PATH):
    loaded_model = safe_load_joblib(MODEL_PATH)
    if loaded_model is None:
        st.error(f"Failed to load model at {MODEL_PATH}")
else:
    st.warning("Model file not found in models/ (expected best_model_*.joblib). App will run in fallback mode if possible.")

if os.path.exists(preprocessor_path):
    preprocessor = safe_load_joblib(preprocessor_path)
    if preprocessor is None:
        st.error(f"Failed to load preprocessor at {preprocessor_path}")
else:
    st.warning("Preprocessor file not found at models/preprocessing_pipeline.joblib. App will attempt to create placeholders.")

# Determine expected feature size if we have model
expected_dim = None
if loaded_model is not None:
    expected_dim = get_model_expected_dim(loaded_model)
    if expected_dim is None:
        st.warning("Could not determine expected input feature size from the model. The app will attempt to pad/truncate.")

# If final_model_input.pkl is present we can show its feature size
bundle_info = None
bundle_path = "data/processed/final_model_input.pkl"
if os.path.exists(bundle_path):
    bundle_info = safe_load_joblib(bundle_path)

if expected_dim is None and bundle_info is not None:
    try:
        expected_dim = int(bundle_info["X"].shape[1])
        st.info(f"Using feature dimension from data/processed/final_model_input.pkl: {expected_dim}")
    except Exception:
        pass

st.write("---")
st.subheader("Input listing")
with st.form("input_form"):
    title = st.text_input("Listing Title", "")
    description = st.text_area("Description", "")
    address = st.text_input("Address", "")
    city = st.text_input("City", "")
    num_of_bedrooms = st.number_input("Bedrooms", min_value=0, max_value=20, value=1, step=1)
    num_of_bathrooms = st.number_input("Bathrooms", min_value=0, max_value=10, value=1, step=1)
    floor_area = st.number_input("Floor area (sqm)", min_value=1, max_value=10000, value=50, step=1)
    uploaded_images = st.file_uploader(
        "Upload interior image(s) (optional)", type=["jpg", "jpeg", "png"], accept_multiple_files=True
    )
    prefer_embedding = st.selectbox(
        "If available, prefer embeddings:",
        options=["MiniLM + EfficientNet (training-like)", "CLIP (fallback)"],
    )
    run_button = st.form_submit_button("Predict")


# -----------------------------
# Helper to create the same engineered row used in your FE pipeline
# -----------------------------

def create_full_feature_row(title, description, address, city, bedrooms, bathrooms, floor_area, num_images):
    full_text = f"{title}. {description}" if (title or description) else ""
    row = {
        "id": 0,
        "title": title,
        "price": 0,
        "address": address,
        "num_of_bedrooms": bedrooms,
        "num_of_bathrooms": bathrooms,
        "floor_area": floor_area,
        "description": description,
        "list_of_amenities": "",
        "image_urls": "",
        "property_url": "",
        "local_images": "",
        "city": city,
        "full_text": full_text,
        "log_price": 0,
        "title_length": len(title) if title is not None else 0,
        "description_length": len(description) if description is not None else 0,
        "amenities_parsed": "",
        "amenity_count": 0,
        "num_images": num_images,
        "title_word_count": len(title.split()) if title else 0,
        "description_word_count": len(description.split()) if description else 0,
        "address_length": len(address) if address else 0,
    }
    return pd.DataFrame([row])


# -----------------------------
# Embedding functions
# -----------------------------

def get_text_embedding(text, prefer="MiniLM"):
    if prefer.startswith("MiniLM") and text_encoder is not None:
        try:
            return np.asarray(text_encoder.encode(text))
        except Exception:
            pass
    if clip_encoder is not None:
        try:
            return np.asarray(clip_encoder.encode(text))
        except Exception:
            pass
    return None


def get_image_embedding_pytorch(img_pil):
    if efficientnet_model is None or img_transform is None:
        return None
    try:
        img = img_pil.convert("RGB")
        tensor = img_transform(img).unsqueeze(0)
        with torch.no_grad():
            feats = efficientnet_model(tensor).squeeze().cpu().numpy()
        return feats
    except Exception:
        return None


def get_image_embedding_clip(img_pil):
    if clip_encoder is None:
        return None
    try:
        # sentence-transformers CLIP supports PIL images as input for encode
        return np.asarray(clip_encoder.encode(img_pil))
    except Exception:
        return None


# -----------------------------
# Prediction flow
# -----------------------------
if run_button:
    st.info("Preparing features...")

    num_images = len(uploaded_images) if uploaded_images else 0

    # create engineered row
    df_row = create_full_feature_row(
        title, description, address, city, num_of_bedrooms, num_of_bathrooms, floor_area, num_images
    )

    # Try to transform tabular features using preprocessor
    X_tab = None
    tab_len = None
    try:
        if preprocessor is not None:
            X_tab_sparse = preprocessor.transform(df_row)
            try:
                X_tab = X_tab_sparse.toarray()
            except Exception:
                X_tab = np.asarray(X_tab_sparse)
            X_tab = ensure_2d(X_tab)
            tab_len = X_tab.shape[1]
            st.success(f"Tabular features prepared (length={tab_len}).")
        else:
            st.warning("Preprocessor unavailable — will infer/construct a zero tabular vector (fallback).")
            X_tab = None
    except Exception as e:
        st.warning(f"Preprocessor.transform failed: {e}. Will try placeholder tabular vector.")
        X_tab = None

    # Text embedding
    text_vec = None
    text_embedding_used = None
    try:
        if prefer_embedding.startswith("MiniLM") and text_encoder is not None:
            text_vec = get_text_embedding(df_row["full_text"].iloc[0], prefer="MiniLM")
            if text_vec is not None:
                text_embedding_used = "MiniLM (384)"
        if text_vec is None and clip_encoder is not None:
            text_vec = get_text_embedding(df_row["full_text"].iloc[0], prefer="CLIP")
            if text_vec is not None:
                text_embedding_used = "CLIP (512)"
    except Exception:
        text_vec = None

    if text_vec is None:
        st.warning("Could not compute text embedding. Using zero-vector fallback.")

    # Image embeddings: average over uploaded images if any
    img_vec = None
    image_embedding_used = None
    if uploaded_images and len(uploaded_images) > 0:
        img_feats = []
        for f in uploaded_images:
            try:
                img_pil = Image.open(f).convert("RGB")
            except Exception:
                continue
            if prefer_embedding.startswith("MiniLM") and efficientnet_model is not None:
                v = get_image_embedding_pytorch(img_pil)
                if v is not None:
                    img_feats.append(v)
                    image_embedding_used = "EfficientNet-B0 (feature)"
                    continue
            if clip_encoder is not None:
                v = get_image_embedding_clip(img_pil)
                if v is not None:
                    img_feats.append(v)
                    image_embedding_used = "CLIP (feature)"
                    continue
        if img_feats:
            img_vec = np.mean(img_feats, axis=0)
        else:
            st.warning("Could not compute any image embeddings — using zero-vector fallback.")
    else:
        st.info("No images uploaded — using zero-vector for images.")

    # If text or image embedding None -> set zero vectors of reasonable default dims
    if text_vec is None:
        if text_encoder is not None:
            text_vec = np.zeros(384, dtype=float)
            text_embedding_used = text_embedding_used or "MiniLM (zero)"
        elif clip_encoder is not None:
            text_vec = np.zeros(512, dtype=float)
            text_embedding_used = text_embedding_used or "CLIP (zero)"
        else:
            text_vec = np.zeros(384, dtype=float)
            text_embedding_used = text_embedding_used or "zero(384)"

    if img_vec is None:
        if efficientnet_model is not None:
            img_vec = np.zeros(1280, dtype=float)
            image_embedding_used = image_embedding_used or "EfficientNet (zero)"
        elif clip_encoder is not None:
            img_vec = np.zeros(512, dtype=float)
            image_embedding_used = image_embedding_used or "CLIP (zero)"
        else:
            img_vec = np.zeros(1280, dtype=float)
            image_embedding_used = image_embedding_used or "zero(1280)"

    # coerce to numpy arrays and ensure 2D
    text_vec = ensure_2d(np.asarray(text_vec))
    img_vec = ensure_2d(np.asarray(img_vec))

    # If X_tab not created, make zero vector of length inferred
    if X_tab is None:
        inferred_tab_len = None
        if preprocessor is not None:
            try:
                dummy = create_full_feature_row("", "", "", "", 0, 0, 0, 0)
                xd = preprocessor.transform(dummy)
                try:
                    inferred_tab_len = xd.toarray().shape[1]
                except Exception:
                    inferred_tab_len = np.asarray(xd).shape[1]
            except Exception:
                inferred_tab_len = None
        if inferred_tab_len is None and bundle_info is not None:
            try:
                total_dim = int(bundle_info["X"].shape[1])
                try_tab = total_dim - (384 + 1280)
                if try_tab > 0:
                    inferred_tab_len = try_tab
            except Exception:
                pass
        if inferred_tab_len is None:
            st.warning("Unable to infer tabular vector length. Using zero-length tabular vector as fallback.")
            X_tab = np.zeros((1, 0))
            tab_len = 0
        else:
            X_tab = np.zeros((1, int(inferred_tab_len)))
            tab_len = int(inferred_tab_len)
            st.info(f"Using zero tabular vector of length {tab_len} (inferred).")
    else:
        X_tab = ensure_2d(np.asarray(X_tab))
        tab_len = X_tab.shape[1]

    st.write(f"Tabular len: {tab_len}, Text emb len: {text_vec.shape[1]}, Image emb len: {img_vec.shape[1]}")
    st.write(f"Text encoder used: {text_embedding_used}, Image encoder used: {image_embedding_used}")

    # Combine
    try:
        X_combined = np.hstack([X_tab, text_vec, img_vec])
    except Exception as e:
        st.error("Failed to concatenate features: " + str(e))
        st.stop()

    total_len = X_combined.shape[1]
    st.write(f"Combined feature length = {total_len}")

    final_vec = X_combined.ravel()
    if expected_dim is not None:
        if total_len != expected_dim:
            
            final_vec = pad_or_truncate(final_vec, expected_dim)
    else:
        if bundle_info is not None and "X" in bundle_info:
            try:
                expected_dim2 = int(bundle_info["X"].shape[1])
                if total_len != expected_dim2:
                   
                    final_vec = pad_or_truncate(final_vec, expected_dim2)
            except Exception:
                pass

    X_final = final_vec.reshape(1, -1)

    # Predict
    if loaded_model is not None:
        try:
            pred_log = float(loaded_model.predict(X_final)[0])
            pred_price = float(np.expm1(pred_log))
            st.success(f"Estimated rental price: **{pred_price:,.0f}** currency units")
            st.write(f"(log-price prediction = {pred_log:.4f})")
        except Exception as e:
            st.error("Model prediction failed: " + str(e))
            st.write(traceback.format_exc())
            st.info("You can re-upload your model + preprocessor into /models and retry.")
    else:
        st.info("Model is not loaded — showing fallback estimate (mean of training set if available).")
        if bundle_info is not None and "y" in bundle_info:
            try:
                mean_log = float(np.mean(bundle_info["y"]))
                pred_price = float(np.expm1(mean_log))
                st.success(f"Fallback estimate (mean): **{pred_price:,.0f}** currency units")
            except Exception:
                st.info("No fallback y available in bundle. Returning dummy 0.")
                st.success(f"Fallback estimate: **0**")
        else:
            st.success("Fallback estimate: **0**")

st.write("---")
st.caption(
    "Notes: if you re-upload your trained model (models/best_model_*.joblib) and preprocessor (models/preprocessing_pipeline.joblib) this app will use them exactly. Padding/truncating is a temporary fallback to prevent crashes; for best results re-upload your original artifacts."
)
