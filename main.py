#!/usr/bin/env python3
"""
Complete Sentiment Analysis Pipeline
Integrates: Data Preprocessing + Model Training + MLflow + DVC
"""

import argparse
import os
from generate_sample_data import *
from preprocessing import TextPreprocessor
from train import ModelTrainer, train_all_models
from dvc_setup import DVCManager

def run_complete_pipeline(data_path=None, models=['lstm', 'gru', 'textcnn'], setup_dvc=True):
    """
    Run the complete sentiment analysis pipeline
    
    Args:
        data_path: Path to input CSV file (optional, uses config default if None)
        models: List of models to train
        setup_dvc: Whether to setup DVC tracking
    """
    
    print("="*70)
    print("🚀 SENTIMENT ANALYSIS PIPELINE")
    print("="*70)
    
    # Step 1: Generate sample data (if needed)
    if not os.path.exists('data/raw/sentiment_data.csv'):
        print("\n📊 Step 1: Generating sample data...")
        sample_data = {
            'message': [
                'I absolutely love this product! It exceeded my expectations.',
                'Terrible experience. Would not recommend to anyone.',
                'The service was okay, nothing special.',
                'Best purchase I have ever made! Highly satisfied.',
                'Worst quality ever. Complete waste of money.',
                'It is fine. Does what it is supposed to do.',
                'Amazing quality and fast delivery! Very happy.',
                'Poor customer service and defective product.',
                'Average product at an average price point.',
                'Outstanding! Will definitely buy again soon.'
            ],
            'sentiment': [
                'positive', 'negative', 'neutral', 'positive', 'negative',
                'neutral', 'positive', 'negative', 'neutral', 'positive'
            ]
        }
        
        import pandas as pd
        df = pd.DataFrame(sample_data)
        os.makedirs('data/raw', exist_ok=True)
        df.to_csv('data/raw/sentiment_data.csv', index=False)
        print("✅ Sample data generated!")
    
    # Step 2: Data Preprocessing
    print("\n⚙️ Step 2: Data Preprocessing...")
    preprocessor = TextPreprocessor()
    df = preprocessor.load_data(data_path)
    X_train, X_test, y_train, y_test = preprocessor.preprocess(df)
    preprocessor.save_artifacts(X_train, X_test, y_train, y_test)
    print("✅ Preprocessing complete!")
    
    # Step 3: Setup DVC (optional)
    if setup_dvc:
        print("\n📦 Step 3: Setting up DVC...")
        try:
            dvc_manager = DVCManager()
            dvc_manager.initialize_dvc()
            dvc_manager.setup_remote_storage()
            dvc_manager.track_data()
            dvc_manager.track_processed_data()
            print("✅ DVC setup complete!")
        except Exception as e:
            print(f"⚠️ DVC setup skipped: {e}")
    
    # Step 4: Model Training
    print("\n🏋️ Step 4: Training Models...")
    trainer = ModelTrainer()
    results = {}
    
    for model_name in models:
        print(f"\n{'='*70}")
        print(f"Training {model_name.upper()} Model")
        print(f"{'='*70}")
        
        try:
            model, metrics, history = trainer.train_model(model_name)
            results[model_name] = {
                'model': model,
                'metrics': metrics,
                'history': history
            }
            print(f"✅ {model_name.upper()} training complete!")
        except Exception as e:
            print(f"❌ Error training {model_name}: {e}")
            continue
    
    # Step 5: Track models with DVC
    if setup_dvc and os.path.exists('saved_models'):
        print("\n📦 Step 5: Tracking models with DVC...")
        try:
            dvc_manager.track_models()
            print("✅ Models tracked!")
        except Exception as e:
            print(f"⚠️ Model tracking skipped: {e}")
    
    # Step 6: Results Summary
    print("\n" + "="*70)
    print("📊 RESULTS SUMMARY")
    print("="*70)
    
    if results:
        print(f"\n{'Model':<12} {'Accuracy':<12} {'F1 Score':<12} {'Precision':<12} {'Recall':<12}")
        print("-"*70)
        
        for model_name, result in results.items():
            metrics = result['metrics']
            print(f"{model_name.upper():<12} {metrics['accuracy']:<12.4f} "
                  f"{metrics['f1_score']:<12.4f} {metrics['precision']:<12.4f} "
                  f"{metrics['recall']:<12.4f}")
        
        # Find best model
        best_model = max(results.items(), key=lambda x: x[1]['metrics']['f1_score'])
        print(f"\n🏆 Best Model: {best_model[0].upper()} "
              f"(F1 Score: {best_model[1]['metrics']['f1_score']:.4f})")
    
    print("\n" + "="*70)
    print("✅ PIPELINE COMPLETE!")
    print("="*70)
    print("\n📝 Next Steps:")
    print("1. View MLflow UI: http://localhost:5000")
    print("2. Check saved models: ./saved_models/")
    print("3. View confusion matrices: ./plots/")
    if setup_dvc:
        print("4. Push to DVC remote: dvc push")
        print("5. Commit changes: git add . && git commit -m 'Update models'")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Sentiment Analysis Pipeline')
    parser.add_argument('--data', type=str, default=None, 
                       help='Path to CSV file with message and sentiment columns')
    parser.add_argument('--models', nargs='+', default=['lstm', 'gru', 'textcnn'],
                       choices=['lstm', 'gru', 'textcnn'],
                       help='Models to train (space-separated)')
    parser.add_argument('--no-dvc', action='store_true',
                       help='Skip DVC setup')
    
    args = parser.parse_args()
    
    run_complete_pipeline(
        data_path=args.data,
        models=args.models,
        setup_dvc=not args.no_dvc
    )

if __name__ == "__main__":
    main()
