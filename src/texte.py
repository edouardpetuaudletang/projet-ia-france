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


def corpus_resumes(resumes: pd.DataFrame) -> pd.DataFrame:
    """Corpus analysé dans la partie 4 : résumés français et américains, nettoyés.

    Retire les résumés vides ou trop courts, les articles d'informatique
    quantique (classés à tort en IA par OpenAlex), les articles présents dans
    les deux classements, et les notices bibliographiques données à la place du
    résumé (elles se terminent par une année, comme « … of the ACL. 2023. »).
    """
    r = resumes.dropna(subset=["resume"])
    r = r.loc[r["resume"].str.len() > 200]
    r = r.loc[~r["titre"].fillna("").str.contains(r"\bquantum\b", case=False)]
    communs = r["id"].duplicated(keep=False)          # articles franco-américains
    print(f"Articles présents dans les deux corpus, retirés : {communs.sum() // 2}")
    r = r.loc[~communs]
    notices = r["resume"].str.strip().str.contains(r"\.\s*(?:19|20)\d\d\.?$")   # se termine par « . 2023. »
    print("Notices bibliographiques retirées :", r.loc[notices, "pays"].value_counts().to_dict())
    r = r.loc[~notices].copy()
    r["pays"] = r["pays"].map({"fr": "France", "us": "États-Unis"})
    return r


def frequences(liste_tokens) -> pd.DataFrame:
    """Compte les mots sur tout le corpus : fréquence, rang et part de chaque mot."""
    compte = Counter(tok for tokens in liste_tokens for tok in tokens)
    df = pd.DataFrame(compte.most_common(), columns=["terme", "freq"])
    df["rang"] = np.arange(1, len(df) + 1)
    df["part"] = df["freq"] / df["freq"].sum()
    return df


def comparer_presence(tokens_a, tokens_b, min_articles=20):
    """Compare deux corpus terme par terme.

    Pour chaque terme : la part des articles qui le mentionnent dans chaque
    corpus, puis le rapport entre les deux. Un rapport de 3 veut dire que le
    terme apparaît dans 3 fois plus d'articles (en proportion) dans le corpus A
    que dans le corpus B.

    On compte des articles plutôt que des occurrences : un terme répété vingt
    fois dans un seul résumé ne doit pas peser plus qu'un terme cité dans vingt
    articles différents. On ne garde que les termes présents dans au moins
    `min_articles` articles au total.
    """
    presence_a = Counter(t for tokens in tokens_a for t in set(tokens))
    presence_b = Counter(t for tokens in tokens_b for t in set(tokens))
    t = pd.DataFrame({"articles_a": pd.Series(presence_a), "articles_b": pd.Series(presence_b)}).fillna(0)
    t = t.loc[t["articles_a"] + t["articles_b"] >= min_articles]
    t["part_a"] = t["articles_a"] / len(tokens_a)
    t["part_b"] = t["articles_b"] / len(tokens_b)
    # un article fictif de chaque côté évite les divisions par zéro
    t["rapport"] = ((t["articles_a"] + 1) / (len(tokens_a) + 1)) / ((t["articles_b"] + 1) / (len(tokens_b) + 1))
    return t.rename_axis("terme").sort_values("rapport", ascending=False).reset_index()


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


def tfidf_par_groupe(tokens_par_article, groupes, n=15):
    """Score TF-IDF moyen de chaque terme dans chaque groupe d'articles.

    Méthode du chapitre « bag of words » du cours : chaque résumé est un
    document ; TfidfVectorizer donne à chaque terme un poids d'autant plus fort
    qu'il est fréquent dans le résumé et rare dans l'ensemble du corpus. On fait
    ensuite la moyenne par groupe (pays) et on garde les `n` termes les plus forts.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    textes = [" ".join(tokens) for tokens in tokens_par_article]
    tfidf = TfidfVectorizer(min_df=5)          # termes présents dans au moins 5 résumés
    matrice = tfidf.fit_transform(textes)
    scores = pd.DataFrame(matrice.toarray(), columns=tfidf.get_feature_names_out())
    moyennes = scores.groupby(list(groupes)).mean().T
    return {g: moyennes[g].sort_values(ascending=False).head(n) for g in moyennes.columns}
