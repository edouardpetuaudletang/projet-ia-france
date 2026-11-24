"""Fonctions de graphiques réutilisées par le rapport (une couleur fixe par pays)."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

from style import COULEURS_GROUPES, COULEURS_PAYS

PAYS_ACCENT = {"France", "États-Unis", "Chine"}


def courbes_par_pays(table, titre, ylabel, pourcentage=False, ax=None, fichier=None):
    """Une courbe par colonne (pays) ; la France en trait épais.

    `table` : index = années, colonnes = pays. Les années manquantes restent
    vides (pas de segment tracé par-dessus).
    """
    if ax is None:
        _, ax = plt.subplots()
    for pays in table.columns:
        ax.plot(table.index, table[pays], marker="o", markersize=4,
                color=COULEURS_PAYS.get(pays, "#b4b2ab"), label=pays,
                linewidth=3 if pays == "France" else 1.6,
                zorder=3 if pays == "France" else 2)
    ax.set_title(titre)
    ax.set_ylabel(ylabel)
    if pourcentage:
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.legend(ncol=3, fontsize=9)
    _enregistrer(ax, fichier)
    return ax


def profil_maillons(profil, titre, fichier=None):
    """Part de chaque pays à chaque maillon ; France, États-Unis et Chine en couleur."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(profil.shape[1])
    for pays in profil.index:
        accent = pays in PAYS_ACCENT
        ax.plot(x, profil.loc[pays], marker="o", markersize=7 if accent else 5,
                color=COULEURS_PAYS[pays] if accent else "#c9c8c2",
                linewidth=3 if pays == "France" else 1.5,
                label=pays if accent else None, zorder=3 if accent else 1)
    for xi, v in zip(x, profil.loc["France"]):
        ax.annotate(f"{v:.1%}", (xi, v), textcoords="offset points", xytext=(0, 9),
                    ha="center", color=COULEURS_PAYS["France"], fontsize=9, fontweight="bold")
    ax.set_xticks(x, profil.columns)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_ylabel("Part parmi les 6 pays")
    ax.set_title(titre)
    ax.legend(title="Gris : Royaume-Uni, Allemagne, Canada", fontsize=9)
    _enregistrer(ax, fichier)
    return ax


def barres_rapport(termes, titre, fichier=None):
    """Barres horizontales sur échelle log : rapport des fréquences France / États-Unis.

    Barre vers la droite = terme plus fréquent en France, vers la gauche = aux
    États-Unis. L'échelle log rend symétriques « 3 fois plus » et « 3 fois moins ».
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    termes = termes.sort_values("rapport")
    couleurs = [COULEURS_PAYS["France"] if v > 1 else COULEURS_PAYS["États-Unis"]
                for v in termes["rapport"]]
    ax.barh(termes["terme"], np.log10(termes["rapport"]), color=couleurs, height=0.65)
    ax.axvline(0, color="#8a8984", linewidth=1)
    graduations = [1 / 10, 1 / 3, 1, 3, 10]
    ax.set_xticks(np.log10(graduations), ["÷10", "÷3", "=", "×3", "×10"])
    ax.set_xlabel("Fréquence en France rapportée à la fréquence aux États-Unis")
    ax.set_title(titre)
    ax.grid(axis="y", visible=False)
    _enregistrer(ax, fichier)
    return ax


def nuage_de_points_annote(table, x, y, titre, xlabel, ylabel, fichier=None):
    """Nuage de points avec le nom de chaque ligne écrit à côté de son point."""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(table[x], table[y], s=60, color="#2a78d6", zorder=3)
    # Deux points voisins (Writer et AI21 Labs par exemple) : la seconde étiquette passe à gauche
    etendue_x, etendue_y = np.ptp(table[x]) or 1, np.ptp(table[y]) or 1
    places = []
    for nom, ligne in table.iterrows():
        voisin = any(abs(ligne[x] - px) < 0.15 * etendue_x and abs(ligne[y] - py) < 0.05 * etendue_y
                     for px, py in places)
        ax.annotate(nom, (ligne[x], ligne[y]), textcoords="offset points",
                    xytext=(-6, 4) if voisin else (6, 4), ha="right" if voisin else "left", fontsize=9)
        places.append((ligne[x], ligne[y]))
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(titre)
    _enregistrer(ax, fichier)
    return ax


def _enregistrer(ax, fichier):
    """Ajuste les marges et enregistre la figure si un nom de fichier est donné."""
    ax.figure.tight_layout()
    if fichier is not None:
        ax.figure.savefig(fichier)


def barres_par_pays(serie, titre, xlabel, pourcentage=True, fichier=None):
    """Diagramme en sucette (lollipop) trié, France en couleur, autres pays en gris.

    Le cours recommande la sucette plutôt que la barre : même lecture, moins
    d'encre pour la même information.
    """
    serie = serie.dropna().sort_values()
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(serie) + 1.2))
    couleurs = [COULEURS_PAYS["France"] if p == "France" else "#8a8984" for p in serie.index]
    ax.hlines(serie.index, 0, serie.values, color=couleurs, linewidth=2)
    ax.scatter(serie.values, serie.index, color=couleurs, s=70, zorder=3)
    for y, v in enumerate(serie.values):
        ax.text(v + 0.02 * (serie.max() or 1), y, f"{v:.0%}" if pourcentage else f"{v:.1f}",
                va="center", fontsize=9, color="#52514e")
    if pourcentage:
        ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlim(0, serie.max() * 1.15)
    ax.set_xlabel(xlabel)
    ax.set_title(titre)
    ax.grid(axis="y", visible=False)
    _enregistrer(ax, fichier)
    return ax


def carte_monde(carte, colonne, titre, fichier=None):
    """Carte choroplèthe du monde en classes (discrétisation manuelle).

    Projection Equal Earth : elle respecte les surfaces, contrairement à
    Mercator qui grossit les pays proches des pôles.
    """
    # Classes ]-1 ; 0], ]0 ; 5], ]5 ; 20], ]20 ; 100], ]100 ; +inf[ : la première ne contient que 0
    bornes = [-1, 0, 5, 20, 100, float("inf")]
    etiquettes = ["aucun", "jusqu'à 5", "5 à 20", "20 à 100", "plus de 100"]
    teintes = ListedColormap(["#eeede9", "#c6dbef", "#6baed6", "#2171b5", "#08306b"])
    carte = carte.to_crs("EPSG:8857")
    carte["classe"] = pd.cut(carte[colonne], bins=bornes, labels=etiquettes)
    ax = carte.plot(column="classe", categorical=True, cmap=teintes, legend=True,
                    edgecolor="white", linewidth=0.3, figsize=(11, 5.5),
                    legend_kwds={"title": "Modèles notables", "loc": "lower left",
                                 "fontsize": 9, "frameon": False})
    ax.set_axis_off()
    ax.set_title(titre)
    _enregistrer(ax, fichier)
    return ax


# --------------------------------------------------------------------------
# Figures de la partie 4 (résumés d'articles, France et États-Unis)
# --------------------------------------------------------------------------

def nuage_pays(ax, x, y, pays, titre=None, taille=8):
    """Nuage de points des articles, une couleur par pays, sans axes (ils n'ont pas de sens)."""
    for nom in ["États-Unis", "France"]:
        choisis = (pays == nom).to_numpy()
        ax.scatter(np.asarray(x)[choisis], np.asarray(y)[choisis], s=taille, alpha=0.6,
                   color=COULEURS_PAYS[nom], label=nom)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    if titre:
        ax.set_title(titre)
    return ax


def carte_groupes(table, etiquettes, titre, fichier=None):
    """Carte des articles (colonnes x, y) colorée par groupe, le nom de chaque groupe écrit en son centre."""
    fig, ax = plt.subplots(figsize=(8, 7))
    for g, sous in table.groupby("groupe"):
        ax.scatter(sous["x"], sous["y"], s=8, alpha=0.55, color=COULEURS_GROUPES[g % len(COULEURS_GROUPES)])
    # Étiquettes par-dessus tous les points, au centre de chaque groupe ; si deux centres
    # sont trop proches, la seconde étiquette est remontée pour rester lisible
    centres = {g: (sous["x"].median(), sous["y"].median()) for g, sous in table.groupby("groupe")}
    marge_x, marge_y = 0.30 * np.ptp(table["x"]), 0.05 * np.ptp(table["y"])
    places = []
    for g, (cx, cy) in sorted(centres.items(), key=lambda centre: centre[1][1]):
        while any(abs(cx - px) < marge_x and abs(cy - py) < marge_y for px, py in places):
            cy += marge_y
        places.append((cx, cy))
        ax.annotate(etiquettes[g], (cx, cy), ha="center", va="center",
                    fontsize=8, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9,
                              edgecolor=COULEURS_GROUPES[g % len(COULEURS_GROUPES)]))
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    ax.set_title(titre)
    _enregistrer(ax, fichier)
    return ax


def barres_part_france(groupes, etiquettes, part_corpus, titre, fichier=None):
    """Barres empilées : part des articles français (bleu) et américains (orange) dans chaque groupe.

    `groupes` : une ligne par groupe, colonnes `articles` et `part_france`.
    La ligne pointillée rappelle la part de la France dans tout le corpus.
    """
    t = groupes.sort_values("part_france")
    noms = [f"{etiquettes[g]} ({n})" for g, n in zip(t.index, t["articles"])]
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(t) + 1.6))
    ax.barh(noms, t["part_france"], color=COULEURS_PAYS["France"], height=0.65, label="France")
    ax.barh(noms, 1 - t["part_france"], left=t["part_france"], color=COULEURS_PAYS["États-Unis"],
            height=0.65, label="États-Unis")
    for y, v in enumerate(t["part_france"]):
        ax.text(v - 0.01, y, f"{v:.0%}", va="center", ha="right", color="white", fontsize=9, fontweight="bold")
    ax.axvline(part_corpus, color="#1a1a1a", linestyle="--", linewidth=1)
    ax.text(part_corpus, len(t) - 0.4, f" {part_corpus:.0%} : part de la France dans tout le corpus",
            fontsize=8, va="bottom")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Part des articles du groupe (entre parenthèses : nombre d'articles)")
    ax.set_title(titre, pad=18)
    ax.grid(axis="y", visible=False)
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.22), fontsize=9)
    _enregistrer(ax, fichier)
    return ax


def histogramme_scores(table, colonne, titre, xlabel, fichier=None):
    """Distribution d'un score par pays, sur les mêmes classes, avec la médiane de chaque pays."""
    fig, ax = plt.subplots(figsize=(8, 3.8))
    etendue = (table[colonne].min(), table[colonne].max())   # mêmes classes pour les deux pays
    for pays in ["France", "États-Unis"]:
        valeurs = table.loc[table["pays"] == pays, colonne]
        mediane = f"{valeurs.median():+.3f}".replace(".", ",")
        ax.hist(valeurs, bins=40, range=etendue, alpha=0.55, color=COULEURS_PAYS[pays], density=True,
                label=f"{pays} (médiane {mediane})")
        ax.axvline(valeurs.median(), color=COULEURS_PAYS[pays], linewidth=2)
    ax.set_title(titre)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Densité")
    ax.legend(fontsize=9)
    _enregistrer(ax, fichier)
    return ax


def halteres_groupes(par_groupe, etiquettes, mediane_fr, mediane_us, titre, xlabel, fichier=None):
    """Score médian de chaque pays dans chaque groupe : un point par pays, reliés par un trait.

    `par_groupe` : une ligne par groupe, colonnes `France`, `États-Unis` et `part_france`.
    Les pointillés rappellent la médiane de chaque pays sur tout le corpus.
    """
    t = par_groupe.sort_values("France")
    noms = [f"{etiquettes[g]} ({p:.0%} FR)" for g, p in zip(t.index, t["part_france"])]
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(t) + 1.8))
    ax.hlines(noms, t[["France", "États-Unis"]].min(axis=1), t[["France", "États-Unis"]].max(axis=1),
              color="#b4b2ab", linewidth=2, zorder=1)
    for pays in ["États-Unis", "France"]:
        ax.scatter(t[pays], noms, s=60, color=COULEURS_PAYS[pays], label=pays, zorder=3)
    for pays, mediane in [("France", mediane_fr), ("États-Unis", mediane_us)]:
        ax.axvline(mediane, color=COULEURS_PAYS[pays], linestyle=":", linewidth=1.5, zorder=0)
        ax.text(mediane, len(t) - 0.35, f" médiane {pays}", color=COULEURS_PAYS[pays], fontsize=8,
                ha="left" if pays == "France" else "right", va="bottom")
    ax.set_xlabel(xlabel)
    ax.set_title(titre, pad=30)
    ax.grid(axis="y", visible=False)
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.22), fontsize=9)
    _enregistrer(ax, fichier)
    return ax


def barres_ecarts(ecarts, titre, ylabel, fichier=None):
    """Une barre par variante du score (index de la série), la valeur écrite au-dessus."""
    fig, ax = plt.subplots(figsize=(9, 3.8))
    couleurs = ["#1a1a1a" if "8" in nom else "#7b5ea7" if "théorique" in nom else "#1baf7a" for nom in ecarts.index]
    ax.bar(ecarts.index, ecarts.values, color=couleurs, width=0.65)
    for x, v in enumerate(ecarts.values):
        ax.text(x, v + 0.001, f"{v:.3f}".replace(".", ","), ha="center", fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, ecarts.max() * 1.15)
    ax.set_title(titre)
    ax.set_xticks(range(len(ecarts)), ecarts.index, rotation=30, ha="right", fontsize=9)
    ax.grid(axis="x", visible=False)
    _enregistrer(ax, fichier)
    return ax


def figure_reduction(variance_gardee, acp2, pays, fidelite, fichier=None):
    """Les deux étapes de la carte en trois panneaux : variance gardée par l'ACP,
    carte faite avec l'ACP seule, et part des voisins retrouvés après chaque réduction."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    ax = axes[0]
    ax.plot(range(1, 201), variance_gardee[:200], color="#7b5ea7")
    for k in (2, 50):
        ax.scatter(k, variance_gardee[k - 1], color="#1a1a1a", zorder=3)
        ax.annotate(f"{k} composantes : {variance_gardee[k - 1]:.0%}", (k, variance_gardee[k - 1]),
                    textcoords="offset points", xytext=(10, -12), fontsize=9)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Nombre de composantes gardées (sur 1 024)")
    ax.set_ylabel("Part de la variance conservée")
    ax.set_title("ACP : variance conservée")

    nuage_pays(axes[1], acp2[:, 0], acp2[:, 1], pays, "ACP seule : 2 composantes", taille=5)
    axes[1].legend(markerscale=2, fontsize=9)
    axes[1].set_xlabel("1re composante"); axes[1].set_ylabel("2e composante")

    ax = axes[2]
    ax.barh(fidelite.index, fidelite.values, color=["#b4b2ab", "#b4b2ab", "#7b5ea7"], height=0.55)
    for y, v in enumerate(fidelite.values):
        ax.text(v + 0.02, y, f"{v:.0%}", va="center", fontsize=9)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Part des 10 plus proches voisins retrouvés")
    ax.set_title("La réduction garde-t-elle les voisins ?")
    ax.grid(axis="y", visible=False)

    fig.tight_layout()
    if fichier is not None:
        fig.savefig(fichier)
    return fig
