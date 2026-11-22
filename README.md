# La France est-elle en retard en IA, ou simplement plus transparente ?

Projet Python pour la data science — ENSAE 2A.

## Question

1. **Le retard** : où se place la France dans l'IA, maillon par maillon
   (recherche → modèles → diffusion) ? Décroche-t-elle entre les articles et les modèles ?
2. **L'opacité** : les développeurs publient-ils de moins en moins d'informations
   sur leurs modèles, et l'AI Act (août 2025) a-t-il changé quelque chose ?
3. **La synthèse** : l'opacité des leaders biaise-t-elle la mesure du retard ?

## Sources

| Source | Accès | Usage |
|---|---|---|
| [Epoch AI — Notable AI Models](https://epoch.ai/data/notable_ai_models.csv) | CSV libre | modèles, pays, puissance de calcul, informations publiées ou non |
| [OpenAlex](https://developers.openalex.org) | API, clé gratuite | publications en IA par pays, résumés (NLP) |
| [Hugging Face Hub](https://huggingface.co/docs/huggingface_hub) | API libre | diffusion des modèles ouverts |
| [FMTI, Stanford](https://github.com/stanford-crfm/fmti) | CSV GitHub | score de transparence (validation) |
| [Signataires du code GPAI](https://digital-strategy.ec.europa.eu/en/policies/contents-code-gpai) | page web | avant / après AI Act |

## Reproduire

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
export OPENALEX_API_KEY="..."          # jamais dans le dépôt

python src/collecte_epoch.py           # -> data/raw/epoch_notable_models_<date>.csv
python src/collecte_openalex.py        # -> data/raw/openalex_*.csv
python src/collecte_huggingface.py     # -> data/raw/hf_models_<date>.csv
```

Puis exécuter les notebooks dans l'ordre :

- `notebooks/01_epoch_descriptif.ipynb` — retard et opacité, statistiques descriptives
- `notebooks/02_nlp_termes.ipynb` — de quoi parle la recherche française en IA (nuages de mots, termes surreprésentés, loi de Zipf)

Les fichiers bruts sont datés : une fois collectés, ils sont copiés sur le stockage
MinIO du SSPCloud et les notebooks les relisent depuis là, pour que l'analyse reste
reproductible même si une API change.

## Organisation du dépôt

```
src/           collecte et nettoyage (fonctions réutilisables)
notebooks/     analyses
data/raw/      données brutes datées (non versionnées)
data/processed données nettoyées (non versionnées)
figures/       graphiques exportés pour le rapport
```
