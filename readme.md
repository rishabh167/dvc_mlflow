# 🎯 Sentiment Analysis Pipeline with MLflow & DVC

A complete end-to-end sentiment analysis pipeline with **LSTM**, **GRU**, and **TextCNN** models, integrated with **MLflow** for experiment tracking and **DVC** for data/model versioning.

## 📋 Features

- ✅ **3 Deep Learning Models**: LSTM, GRU, TextCNN
- ✅ **MLflow Integration**: Track experiments, metrics, hyperparameters
- ✅ **DVC Integration**: Version control for data and models
- ✅ **Docker Support**: Containerized environment
- ✅ **Complete Pipeline**: Data preprocessing → Training → Evaluation
- ✅ **Visualization**: Confusion matrices, training plots
- ✅ **Easy Configuration**: YAML-based config

## 🚀 Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or use Docker
docker-compose up -d
```

### 2. Run Complete Pipeline

```bash
# Run entire pipeline with all models
python main.py

# Run specific models only
python main.py --models lstm gru

# Use custom dataset
python main.py --data path/to/your/data.csv
```

### 3. View Results

```bash
# Start MLflow UI
mlflow ui

# Open browser: http://localhost:5000
```

## 📁 Project Structure

```
sentiment-analysis/
├── main.py                    # Complete pipeline orchestration
├── preprocessing.py           # Data preprocessing & feature engineering
├── models.py                  # Model architectures (LSTM, GRU, TextCNN)
├── train.py                   # Training & MLflow tracking
├── dvc_setup.py              # DVC configuration
├── generate_sample_data.py   # Generate sample dataset
├── config.yaml               # Configuration file
├── requirements.txt          # Python dependencies
├── Dockerfile                # Docker configuration
├── docker-compose.yml        # Docker Compose setup
├── data/
│   ├── raw/                  # Raw input data
│   └── processed/            # Preprocessed data
├── saved_models/             # Trained models
├── artifacts/                # Tokenizer, encoders
├── plots/                    # Confusion matrices
└── mlruns/                   # MLflow tracking data
```

## 🔧 Configuration

Edit `config.yaml` to customize:

```yaml
data:
  max_words: 10000        # Vocabulary size
  max_len: 100           # Sequence length
  test_size: 0.2         # Train-test split

models:
  lstm:
    embedding_dim: 128
    lstm_units: 64
    dropout: 0.5
    epochs: 10
    batch_size: 32
```

## 📊 Dataset Format

Your CSV file should have these columns:

```csv
message,sentiment
"I love this product!",positive
"Terrible experience",negative
"It's okay",neutral
```

## 🏋️ Training Individual Models

```bash
# LSTM only
python -c "from train import ModelTrainer; ModelTrainer().train_model('lstm')"

# GRU only
python -c "from train import ModelTrainer; ModelTrainer().train_model('gru')"

# TextCNN only
python -c "from train import ModelTrainer; ModelTrainer().train_model('textcnn')"
```

## 📦 DVC Workflow

```bash
# Initialize DVC
python dvc_setup.py

# Track new data
dvc add data/raw/sentiment_data.csv

# Track models
dvc add saved_models/

# Push to remote
dvc push

# Pull from remote
dvc pull

# Check status
dvc status
```

## 🐳 Docker Usage

```bash
# Build and run
docker-compose up -d

# Access container
docker exec -it sentiment_analysis bash

# Run pipeline inside container
python main.py

# View MLflow UI
# Open: http://localhost:5000
```

## 📈 MLflow Tracking

All experiments are automatically tracked:

- **Parameters**: model type, hyperparameters, config
- **Metrics**: accuracy, precision, recall, F1-score
- **Artifacts**: models, plots, confusion matrices
- **Training curves**: loss and accuracy per epoch

## 🔬 Experiment Comparison

```python
import mlflow

# Set tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

# List all runs
runs = mlflow.search_runs(experiment_names=["sentiment_analysis"])
print(runs[['run_id', 'params.model_type', 'metrics.test_accuracy']])
```

## 🎯 Model Architectures

### LSTM
- Bidirectional LSTM layers
- Dropout regularization
- Dense layers with softmax

### GRU
- Bidirectional GRU layers
- Spatial dropout
- Dense classification head

### TextCNN
- Multiple parallel convolutions (3, 4, 5-grams)
- Max pooling
- Concatenation + Dense layers

## 📊 Evaluation Metrics

For each model, you get:
- Accuracy
- Precision (weighted)
- Recall (weighted)
- F1-Score (weighted)
- Confusion Matrix
- Per-class metrics

## 🔄 Pipeline Steps

1. **Data Generation**: Create sample data or load your CSV
2. **Preprocessing**: 
   - Text cleaning
   - Tokenization
   - Padding
   - Label encoding
3. **DVC Setup**: Version control initialization
4. **Model Training**: Train all selected models
5. **Evaluation**: Calculate metrics and generate plots
6. **MLflow Logging**: Track everything
7. **Model Saving**: Save best models

## 🛠️ Advanced Usage

### Custom Data Path
```bash
python main.py --data /path/to/your/sentiment_data.csv
```

### Skip DVC
```bash
python main.py --no-dvc
```

### Train Specific Models
```bash
python main.py --models lstm textcnn
```

## 📝 Adding Your Own Dataset

1. Prepare CSV with columns: `message`, `sentiment`
2. Update `config.yaml` if needed
3. Run pipeline:
   ```bash
   python main.py --data your_data.csv
   ```

## 🐛 Troubleshooting

**MLflow UI not starting?**
```bash
# Start manually
mlflow ui --host 0.0.0.0 --port 5000
```

**DVC errors?**
```bash
# Reinitialize
rm -rf .dvc
python dvc_setup.py
```

**Import errors?**
```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

## 📚 References

- MLflow: https://mlflow.org/
- DVC: https://dvc.org/
- TensorFlow: https://tensorflow.org/

## 🤝 Contributing

Feel free to open issues or submit pull requests!

## 📄 License

MIT License

---

**Happy Analyzing! 🎉**
