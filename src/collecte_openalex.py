"""Collecte OpenAlex : publications en IA par pays et par an, et résumés.

La clé d'API se lit dans la variable d'environnement OPENALEX_API_KEY
(ou un fichier .env non versionné). Elle ne doit jamais apparaître dans le code.

À vérifier au premier lancement : l'identifiant du sous-domaine
« Artificial Intelligence » dans la classification d'OpenAlex
(SUBFIELD_IA ci-dessous, 1702 a priori). La fonction `chercher_subfield`
permet de le confirmer.
"""

import os
import time

import pandas as pd
import requests
from dotenv import load_dotenv

from config import AUJOURDHUI, CODES_ISO, RAW
from stockage import envoyer

load_dotenv()
API = "https://api.openalex.org"
CLE = os.environ.get("OPENALEX_API_KEY")
SUBFIELD_IA = "1702"
ANNEES = range(2015, 2026)


def _get(endpoint: str, params: dict) -> dict:
    if CLE:
        params = {**params, "api_key": CLE}
    r = requests.get(f"{API}/{endpoint}", params=params, timeout=60)
    r.raise_for_status()
    time.sleep(0.2)  # rester poli avec l'API
    return r.json()


def chercher_subfield(texte: str = "artificial intelligence") -> pd.DataFrame:
    """Affiche les sous-domaines correspondants, pour confirmer SUBFIELD_IA."""
    res = _get("subfields", {"search": texte})["results"]
    return pd.DataFrame([{"id": s["id"], "nom": s["display_name"]} for s in res])


def compter_publications(pays: str, annee: int, tres_citees: bool = False) -> int:
    """Nombre d'articles en IA avec au moins une institution du pays."""
    filtres = [
        f"primary_topic.subfield.id:{SUBFIELD_IA}",
        f"institutions.country_code:{pays}",
        f"publication_year:{annee}",
        "type:article|preprint",
    ]
    params = {"filter": ",".join(filtres), "per_page": 1}
    if tres_citees:
        # approximation : articles dans le 1 % le plus cité de leur année/domaine
        params["filter"] += ",citation_normalized_percentile.is_in_top_1_percent:true"
    return _get("works", params)["meta"]["count"]


def table_publications() -> pd.DataFrame:
    lignes = []
    for nom, code in CODES_ISO.items():
        for annee in ANNEES:
            lignes.append({
                "pays": nom, "annee": annee,
                "publications": compter_publications(code, annee),
                "publications_top1": compter_publications(code, annee, tres_citees=True),
            })
            print(nom, annee, lignes[-1]["publications"])
    return pd.DataFrame(lignes)


def reconstruire_resume(index_inverse: dict | None) -> str:
    """OpenAlex fournit le résumé sous forme {mot: [positions]} : on le remet en ordre."""
    if not index_inverse:
        return ""
    positions = [(pos, mot) for mot, liste in index_inverse.items() for pos in liste]
    return " ".join(mot for _, mot in sorted(positions))


def echantillon_resumes(pays: str, annee_min: int = 2022, n: int = 2000) -> pd.DataFrame:
    """Résumés des articles IA les plus cités d'un pays, pour la partie NLP."""
    filtres = ",".join([
        f"primary_topic.subfield.id:{SUBFIELD_IA}",
        f"institutions.country_code:{pays}",
        f"from_publication_date:{annee_min}-01-01",
        "has_abstract:true",
    ])
    lignes, curseur = [], "*"
    while curseur and len(lignes) < n:
        page = _get("works", {"filter": filtres, "per_page": 200, "cursor": curseur,
                              "sort": "cited_by_count:desc",
                              "select": "id,title,publication_year,cited_by_count,abstract_inverted_index"})
        for w in page["results"]:
            lignes.append({
                "id": w["id"], "titre": w["title"], "annee": w["publication_year"],
                "citations": w["cited_by_count"],
                "resume": reconstruire_resume(w.get("abstract_inverted_index")),
                "pays": pays,
            })
        curseur = page["meta"].get("next_cursor")
    return pd.DataFrame(lignes[:n])


if __name__ == "__main__":
    if not CLE:
        raise SystemExit("Définir OPENALEX_API_KEY avant de lancer la collecte.")
    RAW.mkdir(parents=True, exist_ok=True)
    print(chercher_subfield())
    publications = table_publications()
    publications.to_csv(RAW / f"openalex_publications_{AUJOURDHUI}.csv", index=False)
    print("MinIO ->", envoyer(publications, f"openalex_publications_{AUJOURDHUI}"))
    resumes = pd.concat([echantillon_resumes("fr"), echantillon_resumes("us")])
    resumes.to_csv(RAW / f"openalex_resumes_{AUJOURDHUI}.csv", index=False)
    print("MinIO ->", envoyer(resumes, f"openalex_resumes_{AUJOURDHUI}"))
