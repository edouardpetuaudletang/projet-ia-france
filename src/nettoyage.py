"""Nettoyage de la base Epoch AI et construction des indicateurs."""

import numpy as np
import pandas as pd

from config import DATE_AI_ACT, PAYS_COMPARES, RAW

# Informations dont on mesure la publication (variables d'opacité)
INFOS_PUBLIABLES = {
    "Training compute (FLOP)": "publie_calcul",
    "Parameters": "publie_parametres",
    "Training dataset size (total)": "publie_donnees",
    "Training compute cost (2023 USD)": "publie_cout",
    "Training hardware": "publie_materiel",
}


def charger_dernier(prefixe: str) -> pd.DataFrame:
    """Lit le fichier brut le plus récent dont le nom commence par `prefixe`."""
    fichiers = sorted(RAW.glob(f"{prefixe}_*.csv"))
    if not fichiers:
        raise FileNotFoundError(f"Aucun fichier {prefixe}_*.csv dans {RAW}")
    return pd.read_csv(fichiers[-1])


def _vrai(serie: pd.Series) -> pd.Series:
    """Convertit les colonnes oui/non d'Epoch (formats variables) en booléens."""
    texte = serie.astype(str).str.strip().str.lower()
    return texte.isin({"true", "yes", "1", "open weights (unrestricted)",
                       "open weights (restricted use)", "open weights (non-commercial)"}) \
        | texte.str.startswith("open weights") | texte.str.startswith("yes")


def preparer_epoch(brut: pd.DataFrame) -> pd.DataFrame:
    """Une ligne par modèle, avec dates, indicateurs d'ouverture et d'opacité."""
    df = brut.copy()
    df["date"] = pd.to_datetime(df["Publication date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["annee"] = df["date"].dt.year

    for colonne, indicateur in INFOS_PUBLIABLES.items():
        df[indicateur] = df[colonne].notna() if colonne in df else False
    df["score_transparence"] = df[list(INFOS_PUBLIABLES.values())].sum(axis=1)

    df["ouvert"] = _vrai(df["Open model weights?"]) if "Open model weights?" in df else np.nan
    df["frontiere"] = _vrai(df["Frontier model"]) if "Frontier model" in df else False
    df["apres_ai_act"] = df["date"] >= pd.Timestamp(DATE_AI_ACT)
    df["log10_calcul"] = np.log10(pd.to_numeric(df["Training compute (FLOP)"], errors="coerce"))
    return df


def eclater_pays(df: pd.DataFrame) -> pd.DataFrame:
    """Une ligne par (modèle, pays), avec un poids fractionnaire.

    Un modèle développé par une collaboration franco-américaine compte pour
    0,5 en France et 0,5 aux États-Unis : c'est le comptage fractionnaire,
    choix à justifier dans le rapport.
    """
    pays = (df["Country (of organization)"].fillna("Inconnu")
            .str.split(r"\s*,\s*", regex=True))
    out = df.assign(pays_brut=pays).explode("pays_brut")
    out["poids"] = 1 / out.groupby(level=0)["pays_brut"].transform("count")
    out["pays"] = out["pays_brut"].map(PAYS_COMPARES).fillna("Autres")
    return out.reset_index(drop=True)


def annees_de_retard(df_pays: pd.DataFrame, reference: str = "États-Unis") -> pd.DataFrame:
    """Pour chaque pays et chaque année : depuis quand la référence avait-elle ce niveau ?

    Niveau = puissance de calcul du meilleur modèle connu à date (record cumulé).
    Retard = année - première année où la référence a atteint ce record.
    """
    record = (df_pays.dropna(subset=["log10_calcul"])
              .groupby(["pays", "annee"])["log10_calcul"].max()
              .groupby(level=0).cummax().rename("record").reset_index())
    ref = record.loc[record["pays"] == reference].sort_values("annee")

    def premiere_annee_ref(niveau):
        atteint = ref.loc[ref["record"] >= niveau, "annee"]
        return atteint.min() if not atteint.empty else np.nan

    record["annee_ref_au_meme_niveau"] = record["record"].map(premiere_annee_ref)
    record["retard_annees"] = record["annee"] - record["annee_ref_au_meme_niveau"]
    return record
