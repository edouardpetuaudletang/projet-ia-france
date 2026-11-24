# La France est-elle en retard en IA, ou simplement plus transparente ?

Projet Python pour la data science — ENSAE 2A.

## Question

1. **Le retard** : où se place la France dans l'IA, maillon par maillon
   (recherche → modèles → diffusion) ? Décroche-t-elle entre les articles et les modèles ?
2. **L'opacité** : les développeurs publient-ils de moins en moins d'informations
   sur leurs modèles, et l'AI Act (août 2025) a-t-il changé quelque chose ?
3. **La synthèse** : l'opacité des leaders biaise-t-elle la mesure du retard ?
4. **Le contenu de la recherche** : de quoi parle la recherche française en IA, comparée
   à la recherche américaine ? (analyse des résumés d'articles : spaCy, TF-IDF, nuages de mots)

## Sources

| Source | Accès | Usage |
|---|---|---|
| [Epoch AI — Notable AI Models](https://epoch.ai/data/notable_ai_models.csv) | CSV libre | modèles, pays, puissance de calcul, informations publiées ou non |
| [OpenAlex](https://developers.openalex.org) | API, clé gratuite | publications en IA par pays, résumés (NLP) |
| [Hugging Face Hub](https://huggingface.co/docs/huggingface_hub) | API libre | diffusion des modèles ouverts |
| [FMTI, Stanford](https://github.com/stanford-crfm/fmti) | CSV GitHub | score de transparence (validation) |
| [Signataires du code GPAI](https://digital-strategy.ec.europa.eu/en/policies/contents-code-gpai) | web scraping | avant / après AI Act |
| [Banque mondiale](https://data.worldbank.org) | API libre | population et PIB, pour rapporter les indicateurs à la taille des pays |

## Reproduire

Le rapport final est **`rapport.ipynb`**, à la racine : il s'exécute de bout en bout,
de la lecture des données aux conclusions.

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
export OPENALEX_API_KEY="..."          # clé gratuite, jamais dans le dépôt
```

Par défaut, le rapport relit les données figées (stockage MinIO du SSP Cloud,
sinon `data/raw/`). Mettre `COLLECTER = True` dans sa première cellule de code
relance toute la collecte (`src/pipeline.py`), source par source. Chaque script
de collecte peut aussi être lancé seul, par exemple `python src/collecte_epoch.py`.

Chaque collecte « fige » ses données en Parquet sur MinIO, dans
`<bucket>/diffusion/projet-ia-france/raw/` (lisible par tous les utilisateurs du
SSP Cloud), en plus d'une copie datée dans `data/raw/`. Le bucket se règle avec
la variable d'environnement `PROJET_BUCKET`. Créer au préalable le dossier
`diffusion` via [Mes fichiers](https://datalab.sspcloud.fr/my-files).

Les notebooks de `notebooks/` sont des **annexes** : analyses détaillées et tests
de robustesse qui ne figurent pas dans le rapport.

- `01_epoch_descriptif` — retard et opacité, analyses complètes
- `02_nlp_termes` — analyse textuelle complète (fréquences, loi de Zipf…)
- `03_recherche_modeles` — de la recherche aux modèles, par habitant
- `04_transparence` — Stanford et AI Act en détail

## Organisation du dépôt

```
rapport.ipynb  rapport final
src/           collecte, nettoyage, analyses et graphiques (fonctions réutilisables)
notebooks/     annexes
data/raw/      données brutes datées (non versionnées)
figures/       graphiques exportés pour le rapport
```
