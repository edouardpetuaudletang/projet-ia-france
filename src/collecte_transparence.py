"""Collecte des deux sources sur la transparence des développeurs d'IA.

1. Foundation Model Transparency Index (Stanford CRFM) : pour chaque grand
   développeur, 0/1 sur une centaine d'indicateurs, trois éditions.
   Attention : la liste d'indicateurs a changé en 2025, donc on compare des
   scores agrégés (part d'indicateurs satisfaits), pas indicateur par indicateur.
2. Signataires du code de bonnes pratiques européen (AI Act), extraits de la
   page officielle de la Commission européenne (web scraping).
"""

import re

import pandas as pd
import requests
from bs4 import BeautifulSoup

from config import AUJOURDHUI, RAW
from stockage import envoyer

URL_FMTI = "https://raw.githubusercontent.com/stanford-crfm/fmti/main/{e}/{e}_scores.csv"
EDITIONS = {"October2023": "2023-10", "May2024": "2024-05", "Dec2025": "2025-12"}
URL_CODE = "https://digital-strategy.ec.europa.eu/en/policies/contents-code-gpai"
COLONNES_NON_ENTREPRISE = {"Domain", "Subdomain", "Indicator", "Total"}


def fmti_edition(edition: str) -> pd.DataFrame:
    """Format long : une ligne par (développeur, indicateur)."""
    brut = pd.read_csv(URL_FMTI.format(e=edition))
    brut = brut.loc[~brut["Indicator"].astype(str).str.contains("total", case=False)]
    entreprises = [c for c in brut.columns if c not in COLONNES_NON_ENTREPRISE]
    long = brut.melt(id_vars=["Indicator"] + [c for c in ("Domain", "Subdomain") if c in brut],
                     value_vars=entreprises, var_name="developpeur", value_name="satisfait")
    long["indicateur"] = long["Indicator"].str.strip()
    long["satisfait"] = pd.to_numeric(long["satisfait"], errors="coerce")
    long["edition"] = EDITIONS[edition]
    return long.drop(columns="Indicator")


def fmti() -> pd.DataFrame:
    return pd.concat([fmti_edition(e) for e in EDITIONS], ignore_index=True)


def signataires(url: str = URL_CODE) -> pd.DataFrame:
    """Liste des signataires, telle qu'affichée sur la page de la Commission.

    Le texte de la section « Signatories of the code of practice » est découpé
    ligne par ligne, jusqu'au paragraphe sur les signatures partielles.
    """
    page = requests.get(url, timeout=60, headers={"User-Agent": "projet-ensae"})
    page.raise_for_status()
    texte = BeautifulSoup(page.text, "html.parser").get_text("\n")
    debut = texte.rfind("Signatories of the code of practice")
    fin = texte.find("In addition", debut)
    if debut < 0 or fin < 0:
        raise ValueError("Structure de la page modifiée : vérifier la section des signataires.")
    lignes = [l.strip() for l in texte[debut:fin].split("\n")]
    noms = [l for l in lignes[1:] if l and not l.startswith("Some signatories")]
    df = pd.DataFrame({"organisation": noms, "signature": "complète"})

    # Signatures partielles, décrites en toutes lettres après la liste
    partiel = re.findall(r"In addition, ([\w .,&-]+?) signed up to the ([\w ]+?) Chapter",
                         texte[fin:fin + 1000])
    for nom, chapitre in partiel:
        df.loc[len(df)] = [nom.strip(), f"partielle ({chapitre.strip()})"]
    df["date_collecte"] = AUJOURDHUI
    return df


def main():
    """Collecte, sauvegarde locale datée et copie figée sur MinIO."""
    RAW.mkdir(parents=True, exist_ok=True)
    scores = fmti()
    scores.to_csv(RAW / f"fmti_scores_{AUJOURDHUI}.csv", index=False)
    print("MinIO ->", envoyer(scores, f"fmti_scores_{AUJOURDHUI}"))
    print(scores.groupby(["edition", "developpeur"])["satisfait"].mean().round(2).unstack(0))

    signes = signataires()
    signes.to_csv(RAW / f"signataires_code_{AUJOURDHUI}.csv", index=False)
    print("MinIO ->", envoyer(signes, f"signataires_code_{AUJOURDHUI}"))
    print(signes)


if __name__ == "__main__":
    main()
