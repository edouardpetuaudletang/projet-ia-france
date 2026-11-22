"""Chemins et constantes partagés par tout le projet."""

import os
from datetime import date
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]
RAW = Path(os.environ.get("PROJET_RAW", RACINE / "data" / "raw"))
PROCESSED = RACINE / "data" / "processed"
FIGURES = RACINE / "figures"

AUJOURDHUI = date.today().isoformat()

# Groupe de comparaison retenu (justifié dans le rapport)
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

# Entrée en application des obligations GPAI de l'AI Act
DATE_AI_ACT = "2025-08-02"

# Fond de carte des pays du monde (Natural Earth, domaine public), lu directement par URL
URL_FOND_MONDE = ("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
                  "master/geojson/ne_110m_admin_0_countries.geojson")
