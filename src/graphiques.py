"""Fonctions de graphiques réutilisées par le rapport (une couleur fixe par pays)."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

from style import COULEURS_PAYS

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
