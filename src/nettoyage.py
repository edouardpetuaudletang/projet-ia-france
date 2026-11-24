"""Nettoyage de la base Epoch AI et construction des indicateurs."""

import re

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
    """Lit la version la plus récente d'un fichier brut.

    Ordre de recherche : d'abord la copie figée sur MinIO (source de référence
    pour la reproductibilité), sinon le fichier local dans data/raw.
    """
    try:
        from stockage import lire, lister
        distants = lister(prefixe)
    except Exception:          # hors SSP Cloud ou s3fs absent
        distants = []
    if distants:
        print(f"Lecture MinIO : {distants[-1]}")
        return lire(distants[-1])

    fichiers = sorted(RAW.glob(f"{prefixe}_*.csv"))
    if not fichiers:
        raise FileNotFoundError(f"Aucun fichier {prefixe}_* sur MinIO ni dans {RAW}")
    print(f"Lecture locale : {fichiers[-1]}")
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

    # Ouverture : la colonne « Model accessibility » est la plus complète
    # (Open weights / API access / Unreleased…) ; vide = information inconnue.
    if "Model accessibility" in df:
        acces = df["Model accessibility"].astype("string").str.lower()
        df["ouvert"] = acces.str.startswith("open weights").astype("boolean").where(acces.notna())
    else:
        df["ouvert"] = _vrai(df["Open model weights?"])
    df["frontiere"] = _vrai(df["Frontier model"]) if "Frontier model" in df else False
    df["apres_ai_act"] = df["date"] >= pd.Timestamp(DATE_AI_ACT)
    df["source_calcul"] = qualifier_source_calcul(df)
    df["log10_calcul"] = np.log10(pd.to_numeric(df["Training compute (FLOP)"], errors="coerce"))
    return df


# Notes d'Epoch signalant une puissance de calcul devinée plutôt que calculée
# à partir d'informations publiées (score aux benchmarks, modèles comparables…).
MOTIF_ESTIMATION_INDIRECTE = (r"benchmark|scaling law|peer model|comparable model|"
                              r"guess|speculat|rough|plausibl")


def qualifier_source_calcul(df: pd.DataFrame) -> pd.Series:
    """D'où vient la puissance de calcul d'un modèle dans Epoch ?

    - « non renseignée » : Epoch n'a aucune valeur ;
    - « estimation indirecte » : Epoch l'a devinée (fiabilité « Speculative »,
      ou note qui évoque benchmarks, modèles comparables, ordre de grandeur) ;
    - « calculée depuis des infos publiées » : calcul d'Epoch à partir de ce que
      le développeur a rendu public (paramètres × jetons, matériel × durée…),
      ou valeur déclarée directement.

    Distinction essentielle pour la partie opacité : une valeur renseignée ne
    prouve pas que le développeur l'a divulguée. Sans les colonnes de notes
    (fichier allégé), toute valeur présente est classée dans la 3e catégorie.
    """
    presente = df["Training compute (FLOP)"].notna()
    notes = df.get("Training compute notes", pd.Series("", index=df.index)).fillna("")
    fiabilite = df.get("Confidence", pd.Series("", index=df.index)).fillna("")
    indirecte = (notes.str.contains(MOTIF_ESTIMATION_INDIRECTE, case=False, regex=True)
                 | (fiabilite == "Speculative"))
    return pd.Series(np.select([~presente, indirecte],
                               ["non renseignée", "estimation indirecte"],
                               default="calculée depuis des infos publiées"),
                     index=df.index)


def eclater_pays(df: pd.DataFrame, mode: str = "fractionnaire") -> pd.DataFrame:
    """Une ligne par (modèle, pays), avec un poids.

    - « fractionnaire » (par défaut) : un modèle d'une collaboration
      franco-américaine compte 0,5 pour la France et 0,5 pour les États-Unis ;
      deux organisations américaines comptent 0,5 + 0,5 pour les États-Unis.
    - « entier » : le modèle compte 1 pour chaque pays participant (une seule
      fois par pays). La somme des pays dépasse alors le nombre de modèles.

    Le mode « fractionnaire » est utilisé dans tout le rapport ; le mode « entier »
    sert de test de robustesse.
    """
    if mode not in {"fractionnaire", "entier"}:
        raise ValueError("mode = 'fractionnaire' ou 'entier'")
    pays = (df["Country (of organization)"].fillna("Inconnu")
            .str.split(r"\s*,\s*", regex=True))
    out = df.assign(pays_brut=pays).explode("pays_brut")
    if mode == "fractionnaire":
        out["poids"] = 1 / out.groupby(level=0)["pays_brut"].transform("count")
    else:
        out = out.reset_index().drop_duplicates(["index", "pays_brut"]).set_index("index")
        out["poids"] = 1.0
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
        """Première année où le pays de référence a atteint ce niveau de calcul."""
        atteint = ref.loc[ref["record"] >= niveau, "annee"]
        return atteint.min() if not atteint.empty else np.nan

    record["annee_ref_au_meme_niveau"] = record["record"].map(premiere_annee_ref)
    record["retard_annees"] = record["annee"] - record["annee_ref_au_meme_niveau"]
    return record


# --------------------------------------------------------------------------
# Rapprochement des noms d'organisations entre sources
# --------------------------------------------------------------------------
# Epoch écrit « Google DeepMind », Stanford « Google », la Commission « Google » ;
# Epoch « Meta AI », Stanford « Meta »… On ramène chaque nom à une clé commune.
# La table est volontairement explicite : chaque rapprochement est un choix
# documenté, à compléter si une organisation importante reste non appariée.
CLES_ORGANISATIONS = {
    "deepmind": "Google", "google": "Google",
    "openai": "OpenAI", "anthropic": "Anthropic", "mistral": "Mistral",
    "meta": "Meta", "facebook": "Meta", "microsoft": "Microsoft",
    "amazon": "Amazon", "ibm": "IBM", "xai": "xAI", "alibaba": "Alibaba",
    "qwen": "Alibaba", "deepseek": "DeepSeek", "cohere": "Cohere",
    "aleph alpha": "Aleph Alpha", "writer": "Writer", "ai21": "AI21 Labs",
    "stability": "Stability AI", "nvidia": "NVIDIA", "baidu": "Baidu",
    "black forest labs": "Black Forest Labs", "servicenow": "ServiceNow",
}


def cle_organisation(nom) -> str | None:
    """Clé commune d'une organisation, ou None si elle n'est pas répertoriée."""
    if not isinstance(nom, str):
        return None
    bas = nom.lower()
    for motif, cle in CLES_ORGANISATIONS.items():
        if re.search(rf"(^|[^a-z]){re.escape(motif)}([^a-z]|$)", bas):
            return cle
    return None


def statut_signature(cle, signataires: pd.DataFrame) -> str:
    """'complète', 'partielle (…)' ou 'non signataire' pour une clé d'organisation."""
    if cle is None:
        return "non répertorié"
    table = signataires.assign(cle=signataires["organisation"].map(cle_organisation))
    ligne = table.loc[table["cle"] == cle, "signature"]
    return ligne.iloc[0] if not ligne.empty else "non signataire"
