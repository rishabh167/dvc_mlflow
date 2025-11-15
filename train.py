import mlflow
import mlflow.keras
import yaml
import pickle
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os
from models import SentimentModels
from preprocessing import TextPreprocessor

class ModelTrainer:
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.mlflow_config = self.config['mlflow']
        self.model_builder = SentimentModels(config_path)
        
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(self.mlflow_config['tracking_uri'])
        mlflow.set_experiment(self.mlflow_config['experiment_name'])
        
    def load_data(self):
        """Load preprocessed data"""
        print("📂 Loading preprocessed data...")
        preprocessor = TextPreprocessor()
        return preprocessor.load_artifacts()
    
    def get_callbacks(self):
        """Get training callbacks"""
        return [
            EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=2,
                verbose=1,
                min_lr=1e-6
            )
        ]
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Evaluate model and return metrics"""
        print(f"\n📊 Evaluating {model_name}...")
        
        # Predictions
        y_pred_proba = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        
        # Load label encoder for class names
        with open('artifacts/label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        
        # Classification report
        report = classification_report(
            y_test, y_pred, 
            target_names=label_encoder.classes_,
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'classification_report': report,
            'confusion_matrix': cm
        }
        
        print(f"   ✅ Accuracy: {accuracy:.4f}")
        print(f"   ✅ Precision: {precision:.4f}")
        print(f"   ✅ Recall: {recall:.4f}")
        print(f"   ✅ F1 Score: {f1:.4f}")
        
        return metrics
    
    def plot_confusion_matrix(self, cm, model_name):
        """Plot and save confusion matrix"""
        with open('artifacts/label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=label_encoder.classes_,
                   yticklabels=label_encoder.classes_)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        os.makedirs('plots', exist_ok=True)
        plot_path = f'plots/confusion_matrix_{model_name}.png'
        plt.savefig(plot_path)
        plt.close()
        
        return plot_path
    
    def train_model(self, model_name='lstm'):
        """Train a specific model with MLflow tracking"""
        print(f"\n{'='*60}")
        print(f"🚀 Training {model_name.upper()} Model")
        print(f"{'='*60}")
        
        # Load data
        X_train, X_test, y_train, y_test = self.load_data()
        
        # Get model
        model = self.model_builder.get_model(model_name)
        
        # Get model config
        model_config = self.config['models'][model_name]
        
        # Start MLflow run
        with mlflow.start_run(run_name=f"{model_name}_run") as run:
            print(f"\n📝 MLflow Run ID: {run.info.run_id}")
            
            # Log parameters
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("max_words", self.config['data']['max_words'])
            mlflow.log_param("max_len", self.config['data']['max_len'])
            
            for key, value in model_config.items():
                mlflow.log_param(key, value)
            
            # Train model
            print("\n🏋️ Training model...")
            history = model.fit(
                X_train, y_train,
                validation_split=0.2,
                epochs=model_config['epochs'],
                batch_size=model_config['batch_size'],
                callbacks=self.get_callbacks(),
                verbose=1
            )
            
            # Evaluate model
            metrics = self.evaluate_model(model, X_test, y_test, model_name)
            
            # Log metrics
            mlflow.log_metric("test_accuracy", metrics['accuracy'])
            mlflow.log_metric("test_precision", metrics['precision'])
            mlflow.log_metric("test_recall", metrics['recall'])
            mlflow.log_metric("test_f1_score", metrics['f1_score'])
            
            # Log training history
            for epoch, (loss, acc, val_loss, val_acc) in enumerate(zip(
                history.history['loss'],
                history.history['accuracy'],
                history.history['val_loss'],
                history.history['val_accuracy']
            )):
                mlflow.log_metric("train_loss", loss, step=epoch)
                mlflow.log_metric("train_accuracy", acc, step=epoch)
                mlflow.log_metric("val_loss", val_loss, step=epoch)
                mlflow.log_metric("val_accuracy", val_acc, step=epoch)
            
            # Plot confusion matrix
            cm_path = self.plot_confusion_matrix(metrics['confusion_matrix'], model_name)
            mlflow.log_artifact(cm_path)
            
            # Save model
            os.makedirs('saved_models', exist_ok=True)
            model_path = f'saved_models/{model_name}_model.h5'
            model.save(model_path)
            mlflow.log_artifact(model_path)
            
            # Log model to MLflow
            mlflow.keras.log_model(model, f"{model_name}_model")
            
            print(f"\n✅ {model_name.upper()} training complete!")
            print(f"📦 Model saved to: {model_path}")
            
        return model, metrics, history

def train_all_models():
    """Train all models"""
    trainer = ModelTrainer()
    results = {}
    
    models_to_train = ['lstm', 'gru', 'textcnn']
    
    for model_name in models_to_train:
        try:
            model, metrics, history = trainer.train_model(model_name)
            results[model_name] = {
                'model': model,
                'metrics': metrics,
                'history': history
            }
        except Exception as e:
            print(f"❌ Error training {model_name}: {e}")
            continue
    
    # Compare results
    print(f"\n{'='*60}")
    print("📊 MODEL COMPARISON")
    print(f"{'='*60}")
    print(f"{'Model':<12} {'Accuracy':<12} {'F1 Score':<12} {'Precision':<12} {'Recall':<12}")
    print(f"{'-'*60}")
    
    for model_name, result in results.items():
        metrics = result['metrics']
        print(f"{model_name.upper():<12} {metrics['accuracy']:<12.4f} {metrics['f1_score']:<12.4f} "
              f"{metrics['precision']:<12.4f} {metrics['recall']:<12.4f}")
    
    return results

if __name__ == "__main__":
    # Train all models
    results = train_all_models()
    
    print("\n✅ All models trained successfully!")
    print("🔍 Check MLflow UI at http://localhost:5000")
