# Athena: Sentiment Analysis Training Pipeline

Athena is an end-to-end PyTorch training pipeline for multiclass sentiment analysis. It features automatic dataset downloading, text embeddings caching using HuggingFace's `SentenceTransformers`, model training, evaluation, and comprehensive experiment tracking via MLflow.

## 🚀 Features

- **Automated Data Pipeline**: Automatically downloads and prepares raw dataset CSV files from HuggingFace without polluting your git history.
- **Fast Text Embeddings**: Leverages `sentence-transformers` models (like `all-MiniLM-L6-v2`) to compute representations and caches them locally as PyTorch tensors (`.pt`) for incredibly fast iteration.
- **Config-Driven Training**: All major hyperparameters, data paths, and model configurations are extracted into a central `parameter.yaml` file.
- **MLflow & Databricks Integration**: Automatically tracks experiments, logs metrics (Loss, Accuracy, Final Validation Accuracy), artifacts (Confusion Matrix), and registers models to MLflow/Databricks.
- **ONNX Export**: Trained PyTorch models are seamlessly converted and exported to ONNX format.

## 📁 Project Structure

```text
athena/
├── main.py                   # Main entry point for the training execution
├── parameter.yaml            # Centralized configuration (epochs, batch size, lr, etc.)
├── requirements.txt          # Python dependencies
├── .env                      # Environment variables (Databricks tokens, etc.)
└── src/
    ├── data/                 # Dataset classes, caching logic, and the auto-downloader
    ├── models/               # PyTorch neural network models (Linear, NN, Transformer)
    ├── train/                # Epoch loops, validation loops, and MLflow context managers
    └── utils/                # Evaluation helpers (confusion matrix) and ONNX export logic
```

## 🛠️ Setup & Installation

1. **Clone the repository and enter the directory**:
   ```bash
   git clone <repo-url>
   cd athena
   ```

2. **Set up a Virtual Environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables**:
   Create a `.env` file at the root of the project to configure your Databricks API access for MLflow tracking.
   ```env
   DATABRICKS_HOST=your_databricks_host_url
   DATABRICKS_TOKEN=your_databricks_personal_access_token
   ```

## ⚙️ Configuration

Tune your dataset paths, model architectures, and training hyperparameters in `parameter.yaml`:

```yaml
data:
  train_csv: ".data/train_df.csv"
  ...
embeddings:
  model_name: "all-MiniLM-L6-v2"
  ...
model:
  hidden_dim: 128
training:
  batch_size: 64
  epochs: 10
  learning_rate: 0.01
  ...
```

## 🎮 Usage

Simply run the main script. If the dataset doesn't exist locally in the `.data/` directory, Athena will fetch it automatically before beginning dataset tokenization and embedding.

```bash
python main.py
```

After training concludes, Athena outputs a Confusion Matrix and a Classification Report to your terminal. Full run details, metrics curves, and artifacts are synced to your Databricks MLflow dashboard automatically!
