"""Chemins et constantes partagés par tout le projet."""

import os
from datetime import date
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]
RAW = Path(os.environ.get("PROJET_RAW", RACINE / "data" / "raw"))
FIGURES = RACINE / "figures"

AUJOURDHUI = date.today().isoformat()

# Pays comparés à la France dans tout le projet (noms d'Epoch -> noms en français)
PAYS_COMPARES = {
    "United States of America": "États-Unis",
    "China": "Chine",
    "United Kingdom of Great Britain and Northern Ireland": "Royaume-Uni",
    "United Kingdom": "Royaume-Uni",
    "France": "France",
    "Germany": "Allemagne",
    "Canada": "Canada",
}
CODES_ISO = {"États-Unis": "us", "Chine": "cn", "Royaume-Uni": "gb",
             "France": "fr", "Allemagne": "de", "Canada": "ca"}

# Entrée en application des obligations de l'AI Act pour les modèles d'IA à usage général
DATE_AI_ACT = "2025-08-02"

# Fond de carte des pays du monde (Natural Earth, domaine public). Le fichier est
# versionné dans le dépôt pour que la carte se trace sans connexion ; l'URL est sa source.
FOND_MONDE = RACINE / "data" / "fond_carte" / "ne_110m_admin_0_countries.geojson"
URL_FOND_MONDE = ("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
                  "master/geojson/ne_110m_admin_0_countries.geojson")
