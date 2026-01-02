# Housing Rental Price Estimator

A multi-modal machine learning project that predicts rental prices by combining tabular data, textual descriptions, and interior apartment images.

## 🚀 Overview

This estimator leverages a hybrid architecture to process diverse data types:
- **Tabular Data**: Structured features like number of bedrooms, bathrooms, and floor area.
- **Natural Language**: Textual descriptions processed using state-of-the-art NLP models.
- **Computer Vision**: Interior images analyzed via deep learning feature extractors.

The project is structured to take you from raw data cleaning to a fully functional, interactive web application.

## ✨ Key Features

- **Multimodal Learning**: Integrates tabular, text, and image features into a single predictive model.
- **Advanced NLP**: High-quality text embeddings using `Sentence-Transformers` (`all-MiniLM-L6-v2`).
- **Computer Vision**: Feature extraction using `EfficientNet-B0` and `CLIP`.
- **Flexible Modeling**: Supports powerful gradient boosting frameworks like `XGBoost` and `LightGBM`.
- **Interactive UI**: A user-friendly Streamlit web interface for real-time price predictions.
- **Dockerized Environment**: Fully containerized setup for reproducible research and deployment.

## 🛠️ Technology Stack

- **Core**: Python 3.10, NumPy, Pandas, Scikit-learn.
- **ML Boost**: XGBoost, LightGBM.
- **Deep Learning**: PyTorch, Sentence-Transformers, Torchvision.
- **Web App**: Streamlit.
- **Infrastructure**: Docker, Docker Compose, Makefile.

## 📂 Project Structure

```text
├── data/               # Raw, semi-processed, and final datasets
├── notebooks/          # Step-by-step development pipeline (1-7)
├── src/                # Modular source code
├── streamlite_app/     # Streamlit application source
├── models/             # Saved model artifacts and preprocessors
├── results/            # Performance metrics and visualizations
├── Dockerfile          # Container definition
├── docker-compose.yml  # Multi-container orchestration
└── Makefile            # Automation shortcuts
```

## 🏗️ Getting Started

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) & [Docker Compose](https://docs.docker.com/compose/install/)
- (Optional) Python 3.10+ if running locally without Docker.

### Running with Docker (Recommended)

1.  **Build the containers**:
    ```bash
    make build
    ```
2.  **Start the services**:
    ```bash
    make up
    ```
3.  **Access the interfaces**:
    - **JupyterLab**: [http://localhost:8888](http://localhost:8888) (For development/training)
    - **Streamlit App**: [http://localhost:8501](http://localhost:8501) (For testing the model)

### Local Setup (Alternative)

1.  Create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Run the Streamlit app:
    ```bash
    streamlit run streamlite_app/app.py
    ```

## 📈 Development Pipeline

The project follows a structured workflow documented in the `notebooks/` directory:

1.  `1_cleaning.ipynb`: Data ingestion and cleaning.
2.  `2_EDA.ipynb`: Exploratory Data Analysis.
3.  `3_FEATURE_ENGINEERING.ipynb`: Engineering tabular features.
4.  `4_TEXT_IMAGE_FEATURE_EXTRACTION.ipynb`: Embedding text and extracting image features.
5.  `5_MODEL_TRAINING.ipynb`: Training multimodal XGBoost/LightGBM models.
6.  `6_Evaluation&Packaging.ipynb`: Evaluating performance and exporting artifacts.
7.  `7_Final_Project_Notebook.ipynb`: Consolidated final project overview.

## 📝 License

This project is licensed under the [MIT License](LICENSE).
