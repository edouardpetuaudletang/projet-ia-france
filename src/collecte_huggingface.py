"""Collecte Hugging Face : modèles ouverts des principales organisations.

Hugging Face ne fournit pas le pays : la correspondance organisation -> pays
est construite à la main ci-dessous (et recoupée avec Epoch AI dans le notebook).
"""

import pandas as pd
import requests

from config import AUJOURDHUI, RAW
from stockage import envoyer

ORGANISATIONS = {
    "mistralai": "France", "kyutai": "France", "lightonai": "France",
    "meta-llama": "États-Unis", "google": "États-Unis", "microsoft": "États-Unis",
    "openai": "États-Unis", "nvidia": "États-Unis",
    "Qwen": "Chine", "deepseek-ai": "Chine", "THUDM": "Chine",
    "CohereForAI": "Canada", "stabilityai": "Royaume-Uni",
    "aleph-alpha": "Allemagne",
}


def modeles_organisation(auteur: str, limite: int = 1000) -> pd.DataFrame:
    r = requests.get("https://huggingface.co/api/models",
                     params={"author": auteur, "limit": limite, "sort": "downloads"},
                     timeout=60)
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    if df.empty:
        return df
    df["organisation"] = auteur
    df["pays"] = ORGANISATIONS[auteur]
    return df[["id", "organisation", "pays", "downloads", "likes", "createdAt",
               "pipeline_tag", "tags"]]


def main():
    """Collecte, sauvegarde locale datée et copie figée sur MinIO."""
    RAW.mkdir(parents=True, exist_ok=True)
    hf = pd.concat([modeles_organisation(o) for o in ORGANISATIONS], ignore_index=True)
    hf.to_csv(RAW / f"hf_models_{AUJOURDHUI}.csv", index=False)
    hf["tags"] = hf["tags"].astype(str)   # Parquet n'accepte pas les listes hétérogènes
    print("MinIO ->", envoyer(hf, f"hf_models_{AUJOURDHUI}"))
    print(hf.groupby("pays")["downloads"].sum().sort_values(ascending=False))


if __name__ == "__main__":
    main()
