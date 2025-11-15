import pandas as pd
import numpy as np
import pickle
import re
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

class TextPreprocessor:
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_config = self.config['data']
        self.tokenizer = None
        self.label_encoder = None
        
    def clean_text(self, text):
        """Clean and normalize text"""
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def load_data(self, data_path=None):
        """Load data from CSV"""
        if data_path is None:
            data_path = self.data_config['raw_path']
        
        print(f"📂 Loading data from: {data_path}")
        df = pd.read_csv(data_path)
        print(f"✅ Loaded {len(df)} records")
        return df
    
    def preprocess(self, df):
        """Complete preprocessing pipeline"""
        print("\n🔧 Starting preprocessing...")
        
        # Clean text
        print("1️⃣ Cleaning text...")
        df['cleaned_message'] = df['message'].apply(self.clean_text)
        
        # Remove empty messages
        df = df[df['cleaned_message'].str.len() > 0].reset_index(drop=True)
        print(f"   ✅ {len(df)} records after cleaning")
        
        # Encode labels
        print("2️⃣ Encoding labels...")
        self.label_encoder = LabelEncoder()
        df['label'] = self.label_encoder.fit_transform(df['sentiment'])
        print(f"   ✅ Classes: {list(self.label_encoder.classes_)}")
        
        # Tokenize text
        print("3️⃣ Tokenizing text...")
        self.tokenizer = Tokenizer(num_words=self.data_config['max_words'], oov_token='<OOV>')
        self.tokenizer.fit_on_texts(df['cleaned_message'])
        sequences = self.tokenizer.texts_to_sequences(df['cleaned_message'])
        
        # Pad sequences
        print("4️⃣ Padding sequences...")
        X = pad_sequences(sequences, maxlen=self.data_config['max_len'], padding='post', truncating='post')
        y = df['label'].values
        
        print(f"   ✅ Shape: X={X.shape}, y={y.shape}")
        
        # Train-test split
        print("5️⃣ Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.data_config['test_size'],
            random_state=self.data_config['random_state'],
            stratify=y
        )
        
        print(f"   ✅ Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
        
        return X_train, X_test, y_train, y_test
    
    def save_artifacts(self, X_train, X_test, y_train, y_test):
        """Save preprocessed data and artifacts"""
        print("\n💾 Saving artifacts...")
        
        os.makedirs('data/processed', exist_ok=True)
        os.makedirs('artifacts', exist_ok=True)
        
        # Save data
        data_dict = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        }
        
        with open(self.data_config['processed_path'], 'wb') as f:
            pickle.dump(data_dict, f)
        print(f"   ✅ Data saved to {self.data_config['processed_path']}")
        
        # Save tokenizer
        with open('artifacts/tokenizer.pkl', 'wb') as f:
            pickle.dump(self.tokenizer, f)
        print("   ✅ Tokenizer saved")
        
        # Save label encoder
        with open('artifacts/label_encoder.pkl', 'wb') as f:
            pickle.dump(self.label_encoder, f)
        print("   ✅ Label encoder saved")
        
    def load_artifacts(self):
        """Load saved artifacts"""
        with open('artifacts/tokenizer.pkl', 'rb') as f:
            self.tokenizer = pickle.load(f)
        
        with open('artifacts/label_encoder.pkl', 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        with open(self.data_config['processed_path'], 'rb') as f:
            data_dict = pickle.load(f)
        
        return data_dict['X_train'], data_dict['X_test'], data_dict['y_train'], data_dict['y_test']

if __name__ == "__main__":
    # Run preprocessing
    preprocessor = TextPreprocessor()
    df = preprocessor.load_data()
    X_train, X_test, y_train, y_test = preprocessor.preprocess(df)
    preprocessor.save_artifacts(X_train, X_test, y_train, y_test)
    
    print("\n✅ Preprocessing complete!")
