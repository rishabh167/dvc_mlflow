# 🪟 Windows Quick Start Guide

## The Easiest Way (Recommended)

### Option 1: Use Batch Script
```cmd
run.bat
```
That's it! The script will:
- ✅ Create all directories
- ✅ Install dependencies
- ✅ Run the complete pipeline
- ✅ Skip DVC (to avoid errors)

### Option 2: Use PowerShell Script
```powershell
.\run.ps1
```

### Option 3: Manual Commands
```cmd
# 1. Create directories
mkdir data\raw data\processed saved_models artifacts plots mlruns

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run pipeline (skip DVC)
python main.py --no-dvc

# 4. View results
mlflow ui
```

## 🎯 What You Get

After running any of the above, you'll have:
```
✅ Sample data generated (10 rows)
✅ Data preprocessed
✅ 3 models trained (LSTM, GRU, TextCNN)
✅ Performance metrics calculated
✅ Confusion matrices saved
✅ Models saved in saved_models/
✅ Everything tracked in MLflow
```

## 📊 View Your Results

```cmd
# Start MLflow UI
mlflow ui

# Then open in browser:
# http://localhost:5000
```

## 🔧 If You Want DVC (Optional)

1. **Install Git for Windows**
   - Download: https://git-scm.com/download/win
   - Install with default settings
   - **Restart your terminal/PowerShell**

2. **Run setup**
   ```cmd
   python setup.py
   ```

3. **Run pipeline with DVC**
   ```cmd
   python main.py
   ```

## 📝 Using Your Own Data

```cmd
# Your CSV should have: message, sentiment columns
python main.py --no-dvc --data C:\path\to\your\data.csv

# Or train specific models
python main.py --no-dvc --models lstm gru
```

## ⚡ Common Issues & Fixes

### Issue: Encoding errors with emojis
**Fix:** All scripts have been updated to handle Windows encoding

### Issue: "DVC not found" or "Git not found"
**Fix:** Use `--no-dvc` flag:
```cmd
python main.py --no-dvc
```

### Issue: MLflow UI won't start
**Fix:** Make sure port 5000 is free:
```cmd
netstat -ano | findstr :5000
```

### Issue: Import errors
**Fix:** Reinstall dependencies:
```cmd
pip install -r requirements.txt --upgrade
```

### Issue: CUDA/GPU errors
**Fix:** TensorFlow will automatically use CPU. For GPU:
```cmd
pip install tensorflow-gpu==2.15.0
```

## 🎓 Step-by-Step Explanation

### 1. What Happens When You Run the Pipeline?

**Step 1: Data Generation**
- Creates 10 sample messages with sentiments
- Saves to `data/raw/sentiment_data.csv`

**Step 2: Preprocessing**
- Cleans text (removes special characters)
- Tokenizes (converts words to numbers)
- Pads sequences (makes all same length)
- Saves preprocessed data

**Step 3: Model Training**
- Builds LSTM model
- Trains for 10 epochs
- Evaluates on test data
- Logs to MLflow
- Saves model

- Repeats for GRU model
- Repeats for TextCNN model

**Step 4: Results**
- Generates confusion matrices
- Calculates metrics
- Compares all models
- Shows best model

### 2. Where Are My Files?

```
your_project/
├── data/
│   ├── raw/
│   │   └── sentiment_data.csv          ← Your input data
│   └── processed/
│       └── processed_data.pkl          ← Preprocessed data
│
├── saved_models/
│   ├── lstm_model.h5                   ← Trained LSTM
│   ├── gru_model.h5                    ← Trained GRU
│   └── textcnn_model.h5                ← Trained TextCNN
│
├── artifacts/
│   ├── tokenizer.pkl                   ← Text tokenizer
│   └── label_encoder.pkl               ← Label encoder
│
├── plots/
│   ├── confusion_matrix_lstm.png       ← LSTM confusion matrix
│   ├── confusion_matrix_gru.png        ← GRU confusion matrix
│   └── confusion_matrix_textcnn.png    ← TextCNN confusion matrix
│
└── mlruns/                              ← MLflow experiments
```

### 3. Understanding config.yaml

```yaml
data:
  max_words: 10000      # Vocabulary size (keep top 10k words)
  max_len: 100          # Max sequence length (pad/truncate to 100)
  test_size: 0.2        # 20% data for testing

models:
  lstm:
    epochs: 10          # Number of training iterations
    batch_size: 32      # Samples per training step
    dropout: 0.5        # Dropout rate (prevents overfitting)
```

Want to experiment? Edit `config.yaml` and run again!

## 🚀 Quick Commands Cheat Sheet

```cmd
# Run everything
python main.py --no-dvc

# Run with your data
python main.py --no-dvc --data mydata.csv

# Train only LSTM
python main.py --no-dvc --models lstm

# Train LSTM and GRU only
python main.py --no-dvc --models lstm gru

# View MLflow UI
mlflow ui

# Generate new sample data
python generate_sample_data.py

# Just preprocess data
python preprocessing.py

# Train specific model
python -c "from train import ModelTrainer; ModelTrainer().train_model('lstm')"
```

## 📚 What's Next?

1. **Experiment with hyperparameters**
   - Edit `config.yaml`
   - Change epochs, batch_size, dropout
   - Run again and compare in MLflow

2. **Use your own data**
   - Prepare CSV with `message` and `sentiment` columns
   - Run with `--data your_data.csv`

3. **Deploy the model**
   - Load saved model from `saved_models/`
   - Use for predictions in your app

4. **Compare models**
   - Check MLflow UI
   - Compare accuracy, F1-score
   - Choose best model for production

## 💡 Pro Tips

1. **Start with small epochs** (2-3) for testing
2. **Check MLflow UI** after each run
3. **Keep original data** in a safe place
4. **Backup `saved_models/`** folder
5. **Document your experiments** in MLflow

## ❓ Need Help?

Check the main README.md or QUICKSTART.md for more details!

## 🎉 You're Ready!

Just run:
```cmd
run.bat
```
Or:
```cmd
python main.py --no-dvc
```

Happy Training! 🚀
