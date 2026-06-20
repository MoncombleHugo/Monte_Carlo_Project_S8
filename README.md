# Monte Carlo Project S8

Projet de pricing d'options europeennes par Monte Carlo avec reduction de variance, application Streamlit et notebook d'experiences.

## Contenu

- `pricer_app.py` : application Streamlit de pricing interactif.
- `main_1.ipynb` : notebook de travail pour les comparaisons numeriques et la generation des resultats.
- `main.tex` et `rapport_overleaf_full.tex` : sources LaTeX du rapport.
- `requirements.txt` : dependances Python minimales.
- `.streamlit/config.toml` : configuration locale de l'application Streamlit.

## Fonctionnalites

- Pricing d'une option europeenne 1D avec reference Black-Scholes.
- Pricing de baskets arithmetiques en dimension superieure.
- Comparaison de plusieurs methodes de reduction de variance.
- Etude de sensibilite sur les parametres du modele de Black-Scholes.
- Interface Streamlit pour tester rapidement les methodes et les parametres.

## Installation

Ce projet est pense pour Python 3.10+.

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
```

## Lancer l'application

```powershell
streamlit run pricer_app.py
```

## Utiliser le notebook

Ouvrir `main_1.ipynb` dans VS Code ou Jupyter pour reproduire les tableaux et graphes du projet.

## Reproduire les sorties

Les figures, tableaux et exports generes par le notebook ne sont pas versionnes dans le repo. Ils sont recrees localement dans `figures/` ou a la racine selon la cellule executee.

Si tu veux reconstruire le rapport PDF, genere d'abord les sorties du notebook puis compile `main.tex` ou `rapport_overleaf_full.tex` avec ton outil LaTeX habituel.

## Etat du depot

Le depot est volontairement garde propre: les artefacts de calcul, images exportees et autres fichiers temporaires sont exclus via `.gitignore`.