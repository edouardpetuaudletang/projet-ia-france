"""Collecte OpenAlex : publications en IA par pays et par an, et résumés.

La clé d'API se lit dans la variable d'environnement OPENALEX_API_KEY.

Le sous-domaine « Artificial Intelligence » d'OpenAlex a l'identifiant 1702.
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
    """Envoie une requête à l'API OpenAlex (avec la clé si elle existe) et renvoie le JSON."""
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


# En IA, les actes de conférence (NeurIPS, ICML…) sont le principal canal de
# publication : les exclure ferait disparaître plus de la moitié des articles.
TYPES = "type:article|preprint|conference-paper"


def publications_par_annee(pays: str, tres_citees: bool = False) -> pd.Series:
    """Nombre d'articles en IA par année, avec au moins une institution du pays.

    Une seule requête par pays grâce à `group_by=publication_year`.
    """
    filtres = [
        f"primary_topic.subfield.id:{SUBFIELD_IA}",
        f"institutions.country_code:{pays}",
        f"publication_year:{min(ANNEES)}-{max(ANNEES)}",
        TYPES,
    ]
    if tres_citees:
        filtres.append("citation_normalized_percentile.is_in_top_1_percent:true")
    groupes = _get("works", {"filter": ",".join(filtres), "group_by": "publication_year"})["group_by"]
    return pd.Series({int(g["key"]): g["count"] for g in groupes})


def table_publications() -> pd.DataFrame:
    """Nombre d'articles d'IA (tous, et dans le 1 % les plus cités) par pays et par année."""
    lignes = []
    for nom, code in CODES_ISO.items():
        tout, top = publications_par_annee(code), publications_par_annee(code, tres_citees=True)
        for annee in ANNEES:
            lignes.append({"pays": nom, "annee": annee,
                           "publications": tout.get(annee, 0),
                           "publications_top1": top.get(annee, 0)})
        print(nom, "ok")
    return pd.DataFrame(lignes)


def reconstruire_resume(index_inverse: dict | None) -> str:
    """OpenAlex fournit le résumé sous forme {mot: [positions]} : on le remet en ordre."""
    if not index_inverse:
        return ""
    positions = [(pos, mot) for mot, liste in index_inverse.items() for pos in liste]
    return " ".join(mot for _, mot in sorted(positions))


def echantillon_resumes(pays: str, annee_min: int = 2022, n: int = 1000) -> pd.DataFrame:
    """Résumés des n articles d'IA les plus cités d'un pays, pour la partie NLP.

    Pagination simple (`page=`) : la pagination par curseur exige une clé d'API.
    """
    filtres = ",".join([
        f"primary_topic.subfield.id:{SUBFIELD_IA}",
        f"institutions.country_code:{pays}",
        f"from_publication_date:{annee_min}-01-01",
        "has_abstract:true", TYPES,
    ])
    lignes = []
    for page in range(1, n // 200 + 1):
        res = _get("works", {"filter": filtres, "per_page": 200, "page": page,
                             "sort": "cited_by_count:desc",
                             "select": "id,title,publication_year,cited_by_count,abstract_inverted_index"})
        for w in res["results"]:
            lignes.append({
                "id": w["id"], "titre": w["title"], "annee": w["publication_year"],
                "citations": w["cited_by_count"],
                "resume": reconstruire_resume(w.get("abstract_inverted_index")),
                "pays": pays,
            })
    return pd.DataFrame(lignes)


def main():
    """Collecte, sauvegarde locale datée et copie figée sur MinIO."""
    if not CLE:
        print("Attention : pas de OPENALEX_API_KEY, budget gratuit partagé et limité.")
    RAW.mkdir(parents=True, exist_ok=True)
    print(chercher_subfield())
    publications = table_publications()
    publications.to_csv(RAW / f"openalex_publications_{AUJOURDHUI}.csv", index=False)
    print("MinIO ->", envoyer(publications, f"openalex_publications_{AUJOURDHUI}"))
    resumes = pd.concat([echantillon_resumes("fr"), echantillon_resumes("us")])
    resumes.to_csv(RAW / f"openalex_resumes_{AUJOURDHUI}.csv", index=False)
    print("MinIO ->", envoyer(resumes, f"openalex_resumes_{AUJOURDHUI}"))


if __name__ == "__main__":
    main()
