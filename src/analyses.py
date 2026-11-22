"""Calculs d'agrégats utilisés par le rapport final.

Chaque fonction prend des tables déjà nettoyées (voir nettoyage.py) et renvoie
une table prête à afficher ou à tracer : le notebook ne garde que l'enchaînement
et l'interprétation.
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

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


def temps_de_doublement(epoch_pays):
    """Régression log-linéaire de la puissance de calcul sur la date, par échantillon."""
    calc = epoch_pays.dropna(subset=["log10_calcul"])
    calc = calc.loc[calc["annee"] >= 2015].assign(
        t=lambda d: d["date"].dt.year + d["date"].dt.dayofyear / 365.25)
    echantillons = {"tous les modèles": calc, "modèles de pointe": calc[calc["frontiere"]],
                    "États-Unis": calc[calc["pays"] == "États-Unis"],
                    "Chine": calc[calc["pays"] == "Chine"]}
    lignes = []
    for nom, d in echantillons.items():
        m = smf.ols("log10_calcul ~ t", data=d).fit()
        b, (bas, haut) = m.params["t"], m.conf_int().loc["t"]
        mois = lambda x: 12 * np.log10(2) / x
        lignes.append({"échantillon": nom, "n": int(m.nobs),
                       "doublement (mois)": round(mois(b), 1),
                       "IC 95 % (mois)": f"{mois(haut):.1f} - {mois(bas):.1f}",
                       "R²": round(m.rsquared, 2)})
    return pd.DataFrame(lignes)


def logit_publication(epoch_pays, depuis=2018):
    """Probabilité que la puissance de calcul soit connue : rapports de chances."""
    d = (epoch_pays.loc[(epoch_pays["annee"] >= depuis) & (epoch_pays["poids"] == 1)
                        & epoch_pays["ouvert"].notna()]
         .assign(publie=lambda x: x["publie_calcul"].astype(int),
                 ouvert=lambda x: x["ouvert"].astype(int),
                 frontiere=lambda x: x["frontiere"].astype(int),
                 annees_depuis=lambda x: x["annee"] - depuis))
    m = smf.logit("publie ~ annees_depuis + ouvert + frontiere"
                  " + C(pays, Treatment('États-Unis'))", data=d).fit(disp=False)
    t = pd.DataFrame({"rapport de chances": np.exp(m.params),
                      "IC bas": np.exp(m.conf_int()[0]), "IC haut": np.exp(m.conf_int()[1]),
                      "p-valeur": m.pvalues}).round(3)
    t.index = (t.index.str.replace("C(pays, Treatment('États-Unis'))[T.", "pays : ", regex=False)
               .str.replace("]", "", regex=False))
    return t, m


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
    rho = {"score Epoch complet": t["stanford"].corr(t["score_epoch"], method="spearman"),
           "calcul publié (strict)": t["stanford"].corr(t["calcul_publie_strict"], method="spearman")}
    return t.sort_values("stanford", ascending=False), rho


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
