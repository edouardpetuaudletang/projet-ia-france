"""Embeddings des résumés : chaque résumé devient un vecteur qui résume son sens.

Le TF-IDF compare des mots. Deux résumés qui parlent du même sujet avec des mots
différents (« LLM » et « large language model ») y sont éloignés. Un modèle
d'embedding transforme chaque texte en un vecteur de 1 024 nombres, construit
pour que deux textes de sens proche aient des vecteurs proches (cosinus élevé).

Modèle : microsoft/harrier-oss-v1-0.6b (licence MIT, 0,6 milliard de paramètres),
lancé en local avec la bibliothèque sentence-transformers.

Le calcul prend une quinzaine de minutes sur un ordinateur portable. Il est fait
une seule fois (`python src/embeddings.py`) et les vecteurs sont enregistrés dans
data/processed/. Le rapport relit ce fichier : il n'a besoin ni du modèle ni de
torch pour s'exécuter.
"""

import numpy as np
import pandas as pd

from config import RACINE

PROCESSED = RACINE / "data" / "processed"   # vecteurs calculés une fois, versionnés dans Git

MODELE = "microsoft/harrier-oss-v1-0.6b"
FICHIER_RESUMES = PROCESSED / "embeddings_resumes.parquet"
FICHIER_REFERENCES = PROCESSED / "embeddings_references.parquet"

# Phrases de référence pour situer un résumé entre recherche théorique et
# recherche appliquée. Elles décrivent des types d'articles, sans reprendre les
# mots les plus « français » ou « américains » trouvés par le TF-IDF.
REFERENCES = {
    "théorique": [
        "We prove a theorem and derive convergence guarantees for the algorithm.",
        "We establish theoretical bounds on the generalization error and the sample complexity.",
        "This paper studies the mathematical properties of an optimization problem.",
        "We provide a statistical analysis with formal proofs under mild assumptions.",
    ],
    "appliquée": [
        "We deploy the system in a real-world application and evaluate it with users.",
        "We build a tool that practitioners can use in hospitals, factories or companies.",
        "We release a dataset and a benchmark and compare existing models on it.",
        "We fine-tune a large language model for a specific task and report its accuracy.",
    ],
}


def _encoder(textes: list[str], modele: str = MODELE, appareil: str | None = None) -> np.ndarray:
    """Vecteurs normalisés (norme 1) : le produit scalaire est alors le cosinus."""
    import torch
    from sentence_transformers import SentenceTransformer

    if appareil is None:
        appareil = "mps" if torch.backends.mps.is_available() else "cpu"
    m = SentenceTransformer(modele, device=appareil)
    m = m.float()   # le modèle se charge en bfloat16, environ 25 fois plus lent sur un CPU
    return m.encode(textes, batch_size=8, normalize_embeddings=True, show_progress_bar=True)


def calculer(resumes: pd.DataFrame) -> None:
    """Calcule et enregistre les vecteurs de tous les résumés et des phrases de référence.

    On encode tous les résumés bruts (un par identifiant OpenAlex) : le
    nettoyage du rapport peut ensuite changer sans qu'il faille tout recalculer.
    """
    uniques = resumes.dropna(subset=["resume"]).drop_duplicates("id")
    vecteurs = _encoder(uniques["resume"].tolist())
    sortie = pd.DataFrame(vecteurs.astype(np.float32), index=uniques["id"].values)
    sortie.columns = [f"d{i}" for i in range(sortie.shape[1])]
    PROCESSED.mkdir(parents=True, exist_ok=True)
    sortie.rename_axis("id").reset_index().to_parquet(FICHIER_RESUMES, index=False)

    phrases = [(type_, p) for type_, liste in REFERENCES.items() for p in liste]
    vref = _encoder([p for _, p in phrases])
    ref = pd.DataFrame(vref.astype(np.float32), columns=[f"d{i}" for i in range(vref.shape[1])])
    ref.insert(0, "phrase", [p for _, p in phrases])
    ref.insert(0, "type", [t for t, _ in phrases])
    ref.to_parquet(FICHIER_REFERENCES, index=False)
    print(f"{len(sortie)} résumés -> {FICHIER_RESUMES}")


def charger(ids) -> np.ndarray:
    """Vecteurs des résumés dans l'ordre des identifiants demandés."""
    table = pd.read_parquet(FICHIER_RESUMES).set_index("id")
    return table.loc[list(ids)].to_numpy()


def charger_references() -> pd.DataFrame:
    return pd.read_parquet(FICHIER_REFERENCES)


def score_theorique(vecteurs: np.ndarray, references: pd.DataFrame) -> np.ndarray:
    """Cosinus moyen avec les phrases « théoriques » moins cosinus moyen avec les « appliquées ».

    Positif : le résumé ressemble plus à un article théorique ; négatif : plus à
    un article appliqué. Seul l'écart entre groupes a un sens, pas la valeur
    absolue d'un score.
    """
    colonnes = [c for c in references.columns if c.startswith("d")]
    theo = references.loc[references["type"] == "théorique", colonnes].to_numpy()
    appl = references.loc[references["type"] == "appliquée", colonnes].to_numpy()
    return (vecteurs @ theo.T).mean(axis=1) - (vecteurs @ appl.T).mean(axis=1)


if __name__ == "__main__":
    from nettoyage import charger_dernier
    calculer(charger_dernier("openalex_resumes"))
