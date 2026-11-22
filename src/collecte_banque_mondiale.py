"""Population et PIB des pays comparés (API de la Banque mondiale).

Sert à rapporter publications et modèles à la taille de chaque pays :
comparer la France aux États-Unis en valeur brute n'aurait pas de sens.
"""

import pandas as pd
import requests

from config import AUJOURDHUI, CODES_ISO, RAW
from stockage import envoyer

API = "https://api.worldbank.org/v2/country/{pays}/indicator/{indic}"
INDICATEURS = {"SP.POP.TOTL": "population", "NY.GDP.MKTP.CD": "pib_usd"}


def indicateur(code: str, nom: str) -> pd.DataFrame:
    pays = ";".join(c.upper() for c in CODES_ISO.values())
    r = requests.get(API.format(pays=pays, indic=code),
                     params={"format": "json", "date": "2015:2025", "per_page": 500}, timeout=60)
    r.raise_for_status()
    lignes = r.json()[1]
    iso_vers_nom = {v.upper(): k for k, v in CODES_ISO.items()}
    return pd.DataFrame([{"pays": iso_vers_nom[l["country"]["id"]], "annee": int(l["date"]),
                          nom: l["value"]} for l in lignes])


def table_pays() -> pd.DataFrame:
    tables = [indicateur(c, n) for c, n in INDICATEURS.items()]
    out = tables[0]
    for t in tables[1:]:
        out = out.merge(t, on=["pays", "annee"], how="outer")
    return out.sort_values(["pays", "annee"])


if __name__ == "__main__":
    RAW.mkdir(parents=True, exist_ok=True)
    pays = table_pays()
    pays.to_csv(RAW / f"banque_mondiale_{AUJOURDHUI}.csv", index=False)
    print("MinIO ->", envoyer(pays, f"banque_mondiale_{AUJOURDHUI}"))
    print(pays.tail())
