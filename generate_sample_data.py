import pandas as pd
import os

# Generate sample data
sample_data = {
    'message': [
        'I absolutely love this product! It exceeded my expectations.',
        'Terrible experience. Would not recommend to anyone.',
        'The service was okay, nothing special.',
        'Best purchase I have ever made! Highly satisfied.',
        'Worst quality ever. Complete waste of money.',        'I absolutely love this product! It exceeded my expectations.',
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
        'positive', 'negative', 'neutral', 'positive', 'negative','positive', 'negative', 'neutral', 'positive', 'negative',
        'neutral', 'positive', 'negative', 'neutral', 'positive'
    ]
}

# Create dataframe
df = pd.DataFrame(sample_data)

# Create directory if it doesn't exist
os.makedirs('data/raw', exist_ok=True)

# Save to CSV
df.to_csv('data/raw/sentiment_data.csv', index=False)
print("✅ Sample data generated successfully!")
print(f"📊 Shape: {df.shape}")
print("\n🔍 Sample rows:")
print(df.head())
print(f"\n📈 Sentiment distribution:\n{df['sentiment'].value_counts()}")
