"""Télécharge la base Epoch AI des modèles notables et la date."""

import pandas as pd

from config import AUJOURDHUI, RAW

URL_EPOCH = "https://epoch.ai/data/notable_ai_models.csv"


def telecharger_epoch(url: str = URL_EPOCH) -> pd.DataFrame:
    df = pd.read_csv(url)
    if df.empty:
        raise ValueError("Le fichier Epoch AI est vide : vérifier l'URL.")
    return df


if __name__ == "__main__":
    RAW.mkdir(parents=True, exist_ok=True)
    epoch = telecharger_epoch()
    chemin = RAW / f"epoch_notable_models_{AUJOURDHUI}.csv"
    epoch.to_csv(chemin, index=False)
    print(f"{len(epoch)} modèles, {epoch.shape[1]} colonnes -> {chemin}")
