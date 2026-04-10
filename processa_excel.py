import pandas as pd
import joblib
import sys
import os

def main():
    if len(sys.argv) < 2:
        print("Uso: python processa_excel.py <nome_file_excel_input>")
        return

    input_file = sys.argv[1]

    if not os.path.exists(input_file):
        print(f"Errore: Il file '{input_file}' non esiste.")
        return

    model_filename = 'model.joblib'

    print(f"Caricamento modello da {model_filename}...")
    try:
        pipeline = joblib.load(model_filename)
    except FileNotFoundError:
        print(f"Errore: File del modello '{model_filename}' non trovato. Esegui prima train.py.")
        return

    print(f"Lettura dei dati da {input_file}...")
    try:
        # Legge il file Excel
        df = pd.read_excel(input_file)
    except Exception as e:
        print(f"Errore durante la lettura del file Excel: {e}")
        return

    # Verifica colonne necessarie
    if 'descrizione' not in df.columns or 'luogo' not in df.columns:
        print("Errore: Il file Excel deve contenere le colonne 'descrizione' e 'luogo'.")
        return

    print("Pre-elaborazione dei dati e calcolo delle previsioni...")
    # Pre-elaborazione dati come in addestramento
    df['descrizione'] = df['descrizione'].fillna('')
    df['luogo'] = df['luogo'].fillna('')
    X_new = df['luogo'] + " " + df['descrizione']

    # Fai le previsioni
    predictions = pipeline.predict(X_new)

    # Aggiungi le previsioni al DataFrame
    df['categoria_predetta'] = predictions

    # Salva il risultato
    base_name, ext = os.path.splitext(input_file)
    output_file = f"{base_name}_categorizzato{ext}"

    print(f"Salvataggio dei risultati in {output_file}...")
    try:
        df.to_excel(output_file, index=False)
        print("Fatto! L'elaborazione è completata con successo.")
    except Exception as e:
        print(f"Errore durante il salvataggio del file Excel: {e}")

if __name__ == "__main__":
    main()
