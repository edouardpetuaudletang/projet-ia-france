"""Relance toute la collecte, source par source (appelé depuis rapport.ipynb)."""

import collecte_banque_mondiale
import collecte_epoch
import collecte_huggingface
import collecte_openalex
import collecte_transparence

ETAPES = {
    "Epoch AI": collecte_epoch.main,
    "Hugging Face": collecte_huggingface.main,
    "OpenAlex": collecte_openalex.main,
    "Stanford et signataires du code": collecte_transparence.main,
    "Banque mondiale": collecte_banque_mondiale.main,
}


def collecter_tout():
    """Exécute chaque collecte ; une source en échec n'arrête pas les autres."""
    bilan = {}
    for nom, etape in ETAPES.items():
        try:
            etape()
            bilan[nom] = "ok"
        except Exception as erreur:
            bilan[nom] = f"échec : {type(erreur).__name__} — {erreur}"
        print(f"{nom} : {bilan[nom]}")
    return bilan
