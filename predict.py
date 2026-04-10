import pandas as pd
import joblib

def main():
    model_filename = 'model.joblib'

    print(f"Loading model from {model_filename}...")
    try:
        pipeline = joblib.load(model_filename)
    except FileNotFoundError:
        print(f"Error: Model file '{model_filename}' not found. Please run train.py first.")
        return

    # Simulate new incoming data (can also be loaded from a CSV)
    new_data = pd.DataFrame({
        'descrizione': [
            "Specchietto retrovisore rotto da ignoti",
            "Tamponamento in colonna fermi al semaforo",
            "Macchina grandinata sul cofano e tetto"
        ],
        'luogo': [
            "Via Torino, Milano",
            "Incrocio Viale Europa",
            "Parcheggio di casa"
        ]
    })

    print("\nNew Data to Predict:")
    print(new_data)

    # Preprocess the new data exactly as we did during training
    # Combine the text features
    new_data['descrizione'] = new_data['descrizione'].fillna('')
    new_data['luogo'] = new_data['luogo'].fillna('')
    X_new = new_data['luogo'] + " " + new_data['descrizione']

    # Make predictions
    print("\nMaking predictions...")
    predictions = pipeline.predict(X_new)

    # Add predictions back to the dataframe for display
    new_data['categoria_predetta'] = predictions

    print("\nPrediction Results:")
    print(new_data[['descrizione', 'luogo', 'categoria_predetta']])

if __name__ == "__main__":
    main()
