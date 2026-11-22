"""Calculs d'agrégats utilisés par le rapport final.

Chaque fonction prend des tables déjà nettoyées (voir nettoyage.py) et renvoie
une table prête à afficher ou à tracer : le notebook ne garde que l'enchaînement
et l'interprétation.
"""

import numpy as np
import pandas as pd

from nettoyage import cle_organisation, statut_signature

PAYS = ["France", "États-Unis", "Chine", "Royaume-Uni", "Allemagne", "Canada"]
PERIODES = {"2018-2020": range(2018, 2021), "2021-2023": range(2021, 2024),
            "2024-2025": range(2024, 2026)}


def chaine_par_periode(publications, epoch_pays):
    """Articles, articles très cités, modèles et modèles de pointe par pays et période,
    avec le taux de transformation (modèles notables pour 1 000 articles)."""
    lignes = []
    for nom, annees in PERIODES.items():
        pub = publications.loc[publications["annee"].isin(annees)].groupby("pays")
        mod = epoch_pays.loc[epoch_pays["annee"].isin(annees)]
        t = pd.DataFrame({
            "articles": pub["publications"].sum(),
            "articles_top1": pub["publications_top1"].sum(),
            "modeles": mod.groupby("pays")["poids"].sum(),
            "modeles_pointe": (mod["poids"] * mod["frontiere"]).groupby(mod["pays"]).sum(),
        }).reindex(PAYS).fillna(0)
        t["modeles_pour_1000_articles"] = t["modeles"] / t["articles"] * 1000
        t["periode"] = nom
        lignes.append(t)
    return pd.concat(lignes).rename_axis("pays").reset_index()


def profil_recent(chaine, hf, periode="2024-2025"):
    """Part de chaque pays parmi les six, maillon par maillon, sur une période."""
    r = chaine.loc[chaine["periode"] == periode].set_index("pays")
    telechargements = hf.groupby("pays")["downloads"].sum().reindex(PAYS).fillna(0)
    part = lambda s: s / s.sum() if s.sum() else s
    return pd.DataFrame({
        "Articles": part(r["articles"]),
        "Articles top 1 %": part(r["articles_top1"]),
        "Modèles notables": part(r["modeles"]),
        "Modèles de pointe": part(r["modeles_pointe"]),
        "Téléchargements HF": part(telechargements),
    }).reindex(PAYS)


def record_par_annee(epoch, depuis=2015):
    """Puissance de calcul du modèle le plus puissant de chaque année (en FLOP),
    et son évolution par rapport à l'année précédente."""
    r = epoch.loc[epoch["annee"] >= depuis].dropna(subset=["log10_calcul"])
    # log10_calcul est la version numérique et nettoyée de la colonne FLOP
    record = (10 ** r.groupby("annee")["log10_calcul"].max()).rename("record (FLOP)")
    t = record.to_frame()
    t["× par rapport à l'année précédente"] = t["record (FLOP)"] / t["record (FLOP)"].shift()
    return t


def tableau_publication(epoch, depuis=2018):
    """Part des modèles dont la puissance de calcul est connue,
    par période et selon que les poids du modèle sont ouverts ou non."""
    r = epoch.loc[(epoch["annee"] >= depuis) & epoch["ouvert"].notna()].copy()
    r["periode"] = pd.cut(r["annee"], bins=[depuis - 1, 2020, 2023, 2026],
                          labels=[f"{depuis}-2020", "2021-2023", "2024-2026"])
    r["acces"] = np.where(r["ouvert"].astype(bool), "poids ouverts", "fermé")
    part = r.pivot_table(index="periode", columns="acces", values="publie_calcul",
                         aggfunc="mean", observed=True)
    effectifs = r.pivot_table(index="periode", columns="acces", values="publie_calcul",
                              aggfunc="count", observed=True).add_prefix("nombre, ")
    return part.join(effectifs)


def score_stanford(fmti, signataires):
    """Part des indicateurs de Stanford satisfaits, par développeur et édition."""
    score = (fmti.assign(cle=fmti["developpeur"].map(cle_organisation).fillna(fmti["developpeur"]))
             .groupby(["cle", "edition"])["satisfait"].mean().unstack())
    score["statut"] = [statut_signature(c, signataires) for c in score.index]
    return score


def comparer_mesures(score, epoch, depuis=2024):
    """Score Stanford le plus récent face aux deux mesures tirées d'Epoch."""
    derniere = [c for c in score.columns if c != "statut"][-1]
    recent = epoch.loc[epoch["annee"] >= depuis].assign(cle=lambda d: d["Organization"].map(cle_organisation))
    recent = recent.dropna(subset=["cle"])
    epoch_dev = recent.groupby("cle").agg(
        modeles=("Model", "count"),
        score_epoch=("score_transparence", "mean"),
        calcul_publie_strict=("source_calcul", lambda s: (s == "calculée depuis des infos publiées").mean()))
    t = score[[derniere]].rename(columns={derniere: "stanford"}).join(epoch_dev, how="inner").dropna()
    return t.sort_values("stanford", ascending=False)


def avant_apres_ai_act(epoch, signataires, depuis="2024-01-01"):
    """Part des modèles publiant calcul et données, avant / après août 2025, par groupe."""
    r = epoch.loc[epoch["date"] >= depuis].assign(cle=lambda d: d["Organization"].map(cle_organisation))
    r["statut"] = r["cle"].map(lambda c: statut_signature(c, signataires))
    r["groupe"] = np.where(r["statut"].str.startswith("complète"), "signataire",
                  np.where(r["statut"] == "non signataire", "non signataire", "autre"))
    r = r.loc[r["groupe"] != "autre"]
    r["periode"] = np.where(r["apres_ai_act"], "après août 2025", "avant août 2025")
    return (r.groupby(["groupe", "periode"])
             .agg(modeles=("Model", "count"), publie_calcul=("publie_calcul", "mean"),
                  publie_donnees=("publie_donnees", "mean"))
             .reindex(pd.MultiIndex.from_product([["signataire", "non signataire"],
                                                  ["avant août 2025", "après août 2025"]])))


LABOS_LEADERS = r"OpenAI|Anthropic|Google DeepMind|xAI"


def opacite_par_pays(epoch_pays, depuis=2024):
    """Pour les modèles récents : part dont la puissance de calcul est connue, par pays.

    Pondérée par le comptage fractionnaire, comme les comptes de modèles.
    """
    r = epoch_pays.loc[epoch_pays["annee"] >= depuis]
    poids = r["poids"]
    t = pd.DataFrame({
        "modeles": poids.groupby(r["pays"]).sum(),
        "calcul_connu": (poids * r["publie_calcul"]).groupby(r["pays"]).sum(),
        "ouverts": (poids * r["ouvert"].astype("float").fillna(0)).groupby(r["pays"]).sum(),
    })
    t["part_calcul_connu"] = t["calcul_connu"] / t["modeles"]
    t["part_ouverts"] = t["ouverts"] / t["modeles"]
    return t[["modeles", "part_calcul_connu", "part_ouverts"]].reindex(PAYS + ["Autres"]).dropna(how="all")


def opacite_leaders(epoch, depuis=2024, motif=LABOS_LEADERS):
    """Part des modèles récents dont la puissance de calcul est connue :
    laboratoires de pointe américains contre tous les autres développeurs."""
    r = epoch.loc[epoch["annee"] >= depuis]
    leader = r["Organization"].str.contains(motif, na=False)
    return (r.assign(groupe=np.where(leader, "OpenAI, Anthropic, Google DeepMind, xAI", "autres développeurs"))
             .groupby("groupe").agg(modeles=("Model", "count"),
                                    part_calcul_connu=("publie_calcul", "mean"),
                                    part_pointe=("frontiere", "mean")))
