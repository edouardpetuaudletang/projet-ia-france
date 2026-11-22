"""Fonctions de graphiques réutilisées par le rapport (une couleur fixe par pays)."""

import matplotlib.pyplot as plt
import numpy as np

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


def barres_divergentes(termes, colonne, titre, etiquette, fichier=None):
    """Barres horizontales signées : > 0 associé à la France, < 0 aux États-Unis."""
    fig, ax = plt.subplots(figsize=(8, 7))
    termes = termes.sort_values(colonne)
    couleurs = [COULEURS_PAYS["France"] if v > 0 else COULEURS_PAYS["États-Unis"]
                for v in termes[colonne]]
    ax.barh(termes["terme"], termes[colonne], color=couleurs, height=0.65)
    ax.axvline(0, color="#8a8984", linewidth=1)
    for seuil in (-2, 2):
        ax.axvline(seuil, color="#b4b2ab", linewidth=0.8, linestyle="--")
    ax.set_xlabel(etiquette)
    ax.set_title(titre)
    ax.grid(axis="y", visible=False)
    _enregistrer(ax, fichier)
    return ax


def _enregistrer(ax, fichier):
    ax.figure.tight_layout()
    if fichier is not None:
        ax.figure.savefig(fichier)
