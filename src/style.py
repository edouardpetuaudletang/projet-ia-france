"""Style graphique commun : une couleur fixe par pays, quel que soit le graphique."""

import matplotlib.pyplot as plt

COULEURS_PAYS = {
    "France": "#2a78d6",
    "États-Unis": "#eb6834",
    "Chine": "#1baf7a",
    "Royaume-Uni": "#eda100",
    "Allemagne": "#e87ba4",
    "Canada": "#008300",
    "Autres": "#b4b2ab",
}
ORDRE_PAYS = list(COULEURS_PAYS)


def appliquer_style():
    """Applique le style commun à tous les graphiques (taille, grille, couleurs des axes)."""
    plt.rcParams.update({
        "figure.figsize": (9, 4.8), "figure.dpi": 110,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#e5e4e0", "grid.linewidth": 0.6,
        "axes.edgecolor": "#8a8984", "axes.labelcolor": "#52514e",
        "xtick.color": "#52514e", "ytick.color": "#52514e",
        "axes.titlesize": 12, "axes.titleweight": "bold",
        "lines.linewidth": 2, "legend.frameon": False, "axes.axisbelow": True,
    })
