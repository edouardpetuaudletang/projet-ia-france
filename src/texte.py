"""Outils NLP : nettoyage (spaCy), fréquences, termes surreprésentés, nuages de mots."""

from collections import Counter

import numpy as np
import pandas as pd

# Mots omniprésents dans les résumés scientifiques, sans contenu propre
STOPWORDS_ACADEMIQUES = {
    "paper", "propose", "proposed", "method", "approach", "result", "results",
    "show", "use", "using", "based", "model", "models", "study", "new", "also",
    "work", "present", "performance", "task", "problem", "provide", "however",
    "different", "well", "first", "two", "high", "large", "set",
}


# Corrections de lemmatisation constatées sur le corpus : spaCy lemmatise
# « data » en « datum », et « LLMs » n'est pas ramené à « LLM ».
LEMMES_CORRIGES = {"datum": "data", "llms": "llm"}


def nettoyer(textes, nlp, stop_supp=STOPWORDS_ACADEMIQUES, batch_size=200):
    """Lemmatise et retire stopwords, ponctuation, nombres (méthode du cours)."""
    resultats = []
    for doc in nlp.pipe(textes, batch_size=batch_size, disable=["parser", "ner"]):
        lemmes = [LEMMES_CORRIGES.get(t.lemma_.lower(), t.lemma_.lower()) for t in doc
                  if t.is_alpha and not t.is_stop and len(t) > 2]
        resultats.append([l for l in lemmes if l not in stop_supp])
    return resultats


def frequences(liste_tokens) -> pd.DataFrame:
    compte = Counter(tok for tokens in liste_tokens for tok in tokens)
    df = pd.DataFrame(compte.most_common(), columns=["terme", "freq"])
    df["rang"] = np.arange(1, len(df) + 1)
    df["part"] = df["freq"] / df["freq"].sum()
    return df


def termes_surrepresentes(freq_a: pd.DataFrame, freq_b: pd.DataFrame,
                          alpha: float = 0.01) -> pd.DataFrame:
    """Log-odds ratio avec a priori de Dirichlet (Monroe, Colaresi & Quinn, 2008).

    Compare deux corpus A et B. Le z-score corrige l'avantage des mots rares :
    un z > 2 signale un terme nettement plus fréquent dans A que dans B.
    """
    t = (freq_a.set_index("terme")["freq"].rename("a").to_frame()
         .join(freq_b.set_index("terme")["freq"].rename("b"), how="outer").fillna(0))
    prior = alpha * (t["a"] + t["b"])
    n_a, n_b, a0 = t["a"].sum(), t["b"].sum(), prior.sum()
    lo_a = np.log((t["a"] + prior) / (n_a + a0 - t["a"] - prior))
    lo_b = np.log((t["b"] + prior) / (n_b + a0 - t["b"] - prior))
    delta = lo_a - lo_b
    var = 1 / (t["a"] + prior) + 1 / (t["b"] + prior)
    t["log_odds"] = delta
    t["z"] = delta / np.sqrt(var)
    return t.sort_values("z", ascending=False).reset_index()


def nuage(frequences_dict: dict, ax, titre: str, couleur: str = "#2a78d6"):
    """Nuage de mots : la taille d'un terme est proportionnelle à son poids.

    Une seule couleur par nuage (celle du pays) : la taille porte l'information,
    une variation de couleur suggérerait une seconde dimension qui n'existe pas.
    """
    from wordcloud import WordCloud
    wc = WordCloud(width=900, height=500, background_color="white",
                   color_func=lambda *args, **kwargs: couleur,
                   max_words=80, random_state=21,
                   prefer_horizontal=0.95).generate_from_frequencies(frequences_dict)
    ax.imshow(wc, interpolation="bilinear")
    ax.set_title(titre)
    ax.axis("off")
