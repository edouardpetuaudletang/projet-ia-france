"""Stockage des données brutes sur MinIO (SSP Cloud), au format Parquet.

Conformément aux consignes du cours : les données issues d'API sont « figées »
sur le stockage S3 une fois collectées, et l'analyse relit ces copies.
Le dossier `diffusion` du bucket est lisible par tous les utilisateurs
authentifiés du SSP Cloud, donc par le correcteur.

Les jetons S3 sont injectés automatiquement dans les services du SSP Cloud
(valables 7 jours) : aucun identifiant n'apparaît dans ce code.
"""

import os

import pandas as pd

ENDPOINT = "https://minio.lab.sspcloud.fr"
BUCKET = os.environ.get("PROJET_BUCKET", "epetuaud")   # bucket du membre qui héberge les données
DOSSIER = f"{BUCKET}/diffusion/projet-ia-france/raw"


def _fs():
    import s3fs
    return s3fs.S3FileSystem(client_kwargs={"endpoint_url": ENDPOINT})


def envoyer(df: pd.DataFrame, nom: str) -> str:
    """Écrit `df` en Parquet sur MinIO sous le nom `nom` (sans extension)."""
    chemin = f"{DOSSIER}/{nom}.parquet"
    with _fs().open(chemin, "wb") as f:
        df.to_parquet(f, index=False)
    return chemin


def lister(prefixe: str) -> list[str]:
    """Fichiers Parquet disponibles sur MinIO pour un préfixe, du plus ancien au plus récent."""
    try:
        fichiers = _fs().ls(DOSSIER)
    except FileNotFoundError:
        return []
    return sorted(f for f in fichiers
                  if f.split("/")[-1].startswith(prefixe) and f.endswith(".parquet"))


def lire(chemin: str) -> pd.DataFrame:
    with _fs().open(chemin, "rb") as f:
        return pd.read_parquet(f)
