import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib

def main():
    # Load dataset
    print("Loading data...")
    try:
        df = pd.read_csv('dataset_esempio.csv')
    except FileNotFoundError:
        print("Error: 'dataset_esempio.csv' not found. Please upload your dataset.")
        return

    # Check for required columns
    required_cols = ['descrizione', 'luogo', 'categoria']
    for col in required_cols:
        if col not in df.columns:
            print(f"Error: Missing required column '{col}'.")
            return

    # Combine text features
    # Fill missing values with empty strings
    df['descrizione'] = df['descrizione'].fillna('')
    df['luogo'] = df['luogo'].fillna('')

    # Create a combined feature, e.g., "luogo: [luogo] | descrizione: [descrizione]"
    X = df['luogo'] + " " + df['descrizione']
    y = df['categoria']

    # Split the data into training and validation sets
    print("Splitting data...")
    # Note: For very small datasets, we might skip splitting or reduce test size
    # Adding a check for the size of dataset
    if len(df) < 5:
        print("Dataset too small for train_test_split, training on full dataset...")
        X_train, y_train = X, y
        X_test, y_test = X, y # Eval on train for extremely small datasets
    else:
        # We need at least enough samples for the split to work properly
        # We set stratify=None because some classes might have only 1 sample
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build the machine learning pipeline
    # We use TF-IDF to convert text to numerical vectors, then a Logistic Regression classifier
    print("Building and training model pipeline...")
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ('clf', LogisticRegression(random_state=42, class_weight='balanced'))
    ])

    # Train the model
    pipeline.fit(X_train, y_train)

    # Evaluate the model
    print("Evaluating model...")
    y_pred = pipeline.predict(X_test)
    print("\nClassification Report:")
    # Use zero_division=0 to suppress warnings if some classes are missing in the test set
    print(classification_report(y_test, y_pred, zero_division=0))

    # Save the model
    model_filename = 'model.joblib'
    print(f"Saving model to {model_filename}...")
    joblib.dump(pipeline, model_filename)
    print("Training complete!")

if __name__ == "__main__":
    main()
