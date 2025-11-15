import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Embedding, LSTM, GRU, Dense, Dropout, 
    Conv1D, MaxPooling1D, GlobalMaxPooling1D, 
    Concatenate, Input, SpatialDropout1D
)
import yaml

class SentimentModels:
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_config = self.config['data']
        self.vocab_size = self.data_config['max_words']
        self.max_len = self.data_config['max_len']
        self.num_classes = 3  # positive, negative, neutral
    
    def build_lstm(self):
        """Build LSTM model"""
        model_config = self.config['models']['lstm']
        
        model = Sequential([
            Embedding(self.vocab_size, model_config['embedding_dim'], input_length=self.max_len),
            SpatialDropout1D(0.2),
            LSTM(model_config['lstm_units'], return_sequences=True),
            Dropout(model_config['dropout']),
            LSTM(model_config['lstm_units'] // 2),
            Dropout(model_config['dropout']),
            Dense(32, activation='relu'),
            Dropout(0.3),
            Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def build_gru(self):
        """Build GRU model"""
        model_config = self.config['models']['gru']
        
        model = Sequential([
            Embedding(self.vocab_size, model_config['embedding_dim'], input_length=self.max_len),
            SpatialDropout1D(0.2),
            GRU(model_config['gru_units'], return_sequences=True),
            Dropout(model_config['dropout']),
            GRU(model_config['gru_units'] // 2),
            Dropout(model_config['dropout']),
            Dense(32, activation='relu'),
            Dropout(0.3),
            Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def build_textcnn(self):
        """Build Text CNN model"""
        model_config = self.config['models']['textcnn']
        
        input_layer = Input(shape=(self.max_len,))
        embedding = Embedding(self.vocab_size, model_config['embedding_dim'])(input_layer)
        embedding = SpatialDropout1D(0.2)(embedding)
        
        # Multiple convolution layers with different kernel sizes
        conv_blocks = []
        for kernel_size in model_config['kernel_sizes']:
            conv = Conv1D(
                filters=model_config['filters'],
                kernel_size=kernel_size,
                activation='relu',
                padding='same'
            )(embedding)
            conv = GlobalMaxPooling1D()(conv)
            conv_blocks.append(conv)
        
        # Concatenate all convolution outputs
        concatenated = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]
        
        # Dense layers
        dense = Dense(64, activation='relu')(concatenated)
        dense = Dropout(model_config['dropout'])(dense)
        dense = Dense(32, activation='relu')(dense)
        dense = Dropout(0.3)(dense)
        output = Dense(self.num_classes, activation='softmax')(dense)
        
        model = Model(inputs=input_layer, outputs=output)
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def get_model(self, model_name):
        """Get model by name"""
        models_map = {
            'lstm': self.build_lstm,
            'gru': self.build_gru,
            'textcnn': self.build_textcnn
        }
        
        if model_name not in models_map:
            raise ValueError(f"Model {model_name} not found. Available: {list(models_map.keys())}")
        
        return models_map[model_name]()

if __name__ == "__main__":
    # Test models
    builder = SentimentModels()
    
    print("🏗️ Building LSTM model...")
    lstm_model = builder.build_lstm()
    lstm_model.summary()
    
    print("\n🏗️ Building GRU model...")
    gru_model = builder.build_gru()
    gru_model.summary()
    
    print("\n🏗️ Building TextCNN model...")
    textcnn_model = builder.build_textcnn()
    textcnn_model.summary()
