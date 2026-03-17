# Scoring & Prédiction de Prix Immobilier (Régression)

## 1. Contexte métier et objectif

Le marché immobilier new-yorkais est caractérisé par une grande hétérogénéité de prix. Des transactions à prix anormalement bas (transferts intrafamiliaux, ventes administratives) cohabitent avec des ventes de prestige à plusieurs dizaines de millions de dollars. Dans ce contexte, un investisseur ou un analyste a besoin d'un outil capable de distinguer les **vraies décotes** (opportunités d'achat) des **anomalies de données**.

**Objectif** : construire un modèle de **régression supervisée** qui :

1. **Prédit un prix de vente théorique** d'un bien à partir de ses caractéristiques (borough, type de bien, surface, année de construction, nombre d'unités, etc.).
2. **Compare ce prix estimé au prix réel** de la transaction.
3. **Génère un score de valorisation** :
   - Score > 0 → bien en **surcote** (vendu au-dessus de l'estimation).
   - Score < 0 → bien en **décote** (opportunité d'investissement potentielle).

**Métriques principales** : R² (variance expliquée), MAE / RMSE (erreur absolue), MAPE (erreur relative interprétable métier).

---

## 2. Données et nettoyage

### 2.1 — Source

- **Dataset** : NYC Rolling Sales (`data/nyc-rolling-sales.csv`)
- **Volume** : 84 548 transactions immobilières, 22 colonnes
- **Périmètre** : ventes dans les 5 boroughs de New York City

### 2.2 — Variables

| Type | Nb | Exemples |
|------|----|----------|
| Catégorielles | 11 | `NEIGHBORHOOD`, `BUILDING CLASS CATEGORY`, `TAX CLASS AT PRESENT`, `SALE DATE` |
| Numériques | 10 | `BOROUGH`, `BLOCK`, `LOT`, `ZIP CODE`, `RESIDENTIAL UNITS`, `TOTAL UNITS`, `YEAR BUILT` |
| **Cible** | 1 | `SALE PRICE` |

### 2.3 — Règles de nettoyage

| Étape | Règle | Justification |
|-------|-------|---------------|
| **Prix nuls / ≤ 0** | Suppression des lignes où `SALE PRICE` ≤ 0 | Transferts intrafamiliaux, ventes à $0 (1 154 cas) — ne reflètent pas un prix de marché |
| **Prix aberrants bas** | Exclusion des prix ≤ $1 000 (NB03) | Ventes symboliques à $1–$10 qui faussent les métriques |
| **Winsorization** | Écrêtage aux percentiles P1–P99 (≈ $25 000 – $12,9 M) | Réduction de l'impact des extrêmes sur l'entraînement |
| **Cible manquante** | Suppression (17,36 % de NaN dans `SALE PRICE`) | Impossible de superviser sans cible |
| **Dates** | Parsing de `SALE DATE` → extraction de features temporelles (intervalles de 5 ans) | Capture l'effet du cycle immobilier |
| **Transformation log** | `log1p(SALE PRICE)` appliqué à la cible | Distribution originale très asymétrique (skewness = 23,2) |
| **Outliers (IQR)** | Détection via méthode IQR (Q1 − 1.5×IQR, Q3 + 1.5×IQR) | Diagnostic exploratoire, pas de suppression automatique |

**Impact** : le nettoyage réduit le jeu d'entraînement de 10 000 → 7 002 lignes exploitables.

---

## 3. Installation

### Prérequis

- Python 3.13+
- `pip` ou `conda`

### Mise en place

```bash
# Cloner le projet
git clone <url-du-repo>
cd "Scoring & prédiction de prix immobilier (Régression)"

# Créer et activer l'environnement virtuel
python -m venv env
# Linux / macOS
source env/bin/activate
# Windows (PowerShell)
.\env\Scripts\Activate.ps1

# Installer les dépendances
pip install -r requirements.txt
```

### Dépendances principales

| Catégorie | Packages |
|-----------|----------|
| Data | `numpy`, `pandas`, `scipy` |
| ML | `scikit-learn`, `xgboost`, `optuna` |
| Visualisation | `matplotlib`, `seaborn`, `plotly` |
| Notebooks | `jupyter`, `jupyterlab` |
| Persistance | `joblib` |

---

## 4. Reproduire le pipeline

Exécuter les notebooks **dans l'ordre** :

```
notebooks/01_data_exploration.ipynb   →  EDA
notebooks/02_baseline_models.ipynb    →  Baselines (LR, XGBoost, RF)
notebooks/03_finetuned_model.ipynb    →  Feature Engineering + Finetuning + Stacking + Scoring
```

### Étapes détaillées

| # | Notebook | Action | Sortie |
|---|----------|--------|--------|
| 1 | **01 — EDA** | Analyse exploratoire : distribution des prix, corrélations, test de Kruskal-Wallis sur les catégorielles, détection d'outliers (IQR) | Graphiques + diagnostics statistiques |
| 2 | **02 — Baselines** | Entraînement de 3 modèles (LR, XGBoost, RF) avec hyperparamètres par défaut, évaluation CV 5-folds | `models/baseline_lr.joblib`, `baseline_xgb.joblib`, `baseline_rf.joblib` |
| 3 | **03 — Finetune & Scoring** | Feature engineering (6 variables), GridSearch/RandomizedSearch/Optuna, Stacking, Nested CV, scoring de valorisation | `models/finetuned.joblib` |

**Seed global** : `1204` — utilisé pour le split train/test (`test_size=0.2`) et la cross-validation (`KFold n_splits=5`).

---

## 5. Résultats

### 5.1 — Progression des performances

| Modèle | R² (CV / test) | RMSE | Commentaire |
|--------|----------------|------|-------------|
| Rég. Linéaire (baseline) | 0,10 | 1,71 | Stable mais très faible |
| XGBoost (baseline) | −0,02 | 1,79 | Overfitting sévère |
| Random Forest (baseline) | 0,18 | 1,67 | Meilleur baseline |
| ElasticNet (finetuné) | 0,40 | 0,78 | +0,30 pts grâce au feature engineering |
| XGBoost (Optuna) | 0,63 | 0,62 | Optimisation hyperparamètres |
| Random Forest (Optuna) | 0,63 | 0,62 | Plateau atteint |
| **Stacking (final)** | **R² OOF = 0,64 / R² test = 0,61** | **0,62** | **Modèle retenu** |

### 5.2 — Métriques finales (jeu de test)

| Métrique | Valeur |
|----------|--------|
| **R² (test)** | **0,6073** |
| RMSE (log) | 0,6203 |
| MAE (log) | 0,3985 |
| MAPE | 67,76 % |
| Erreur médiane absolue | $150 019 |
| % prédictions à ±25 % | **50,5 %** |
| % prédictions à ±50 % | **76,0 %** |

### 5.3 — Analyse d'erreurs

- **Par segment de prix** : le modèle est le plus performant sur le **segment standard** ($382K – $1,1M), cœur du marché résidentiel. Le segment bas (< $382K) et le segment luxe (≥ $1,1M) sont moins bien prédits.
- **Sources d'erreur principales** :
  - Transactions atypiques persistantes (ventes entre proches à $25K pour des biens estimés > $1M).
  - Surfaces (`LAND SQUARE FEET`, `GROSS SQUARE FEET`) encodées en texte avec des valeurs invalides — perte d'information.
  - Aucune variable sur l'état du bien, la vue, la proximité des transports.
- **MAPE élevé** (67,76 %) : tiré vers le haut par les biens à bas prix où une petite erreur absolue produit une grande erreur relative.

---

## 6. Stratégie d'utilisation, limites et risques

### 6.1 — Stratégie : top K% des décotes

Le score de valorisation permet de **classer les biens par décote** (prix réel < prix estimé). La stratégie recommandée :

1. **Prédire** le prix théorique de chaque bien avec le modèle Stacking.
2. **Calculer le score** : `score = (prix_réel − prix_estimé) / prix_estimé`.
3. **Trier par score croissant** (décotes les plus fortes en premier).
4. **Cibler le top K%** (ex. top 10 %) pour investigation manuelle par un expert.
5. **Filtrer le segment standard** ($382K – $1,1M) où le modèle est le plus fiable.

> En ciblant le top 10 % des décotes dans le segment standard, l'analyste concentre ses efforts sur les biens présentant le meilleur rapport signal/bruit du modèle.

### 6.2 — Limites

| Limite | Impact |
|--------|--------|
| R² = 0,61 → le modèle n'explique que 61 % de la variance | 39 % de la variance reste inexpliquée → risque de faux signaux |
| Absence de variables qualitatives (état, rénové, vue, étage) | Le modèle ne capture pas les facteurs qualitatifs qui justifient des écarts de prix |
| Transactions atypiques résiduelles | Des ventes non-marchandes persistent malgré le filtrage |
| Modèle entraîné sur données NYC uniquement | Non transférable à d'autres marchés sans ré-entraînement |
| Données statiques (pas de dimension temporelle fine) | Ne capture pas les tendances de marché récentes |

### 6.3 — Risques

- **Faux positifs de décote** : un bien identifié comme « sous-évalué » peut l'être pour des raisons que le modèle ne voit pas (vices cachés, contentieux, location réglementée).
- **Biais de sélection** : le modèle est entraîné sur les transactions effectivement réalisées — les biens retirés du marché (invendus) ne sont pas observés.
- **Utilisation sans expertise** : le scoring ne remplace pas l'expertise d'un professionnel immobilier. Il doit être utilisé comme **outil d'aide à la décision**, pas comme signal d'achat automatique.
- **Dérive temporelle** : les performances se dégraderont si le modèle n'est pas ré-entraîné sur des données récentes.

---

## Artefacts

| Fichier | Description |
|---------|-------------|
| `models/baseline_lr.joblib` | Pipeline Régression Linéaire (baseline) |
| `models/baseline_xgb.joblib` | Pipeline XGBoost (baseline) |
| `models/baseline_rf.joblib` | Pipeline Random Forest (baseline) |
| `models/finetuned.joblib` | **Pipeline Stacking final (production)** |
| `reports/figures/model_report.md` | Rapport technique détaillé |

## Structure du projet

```
├── data/                        # Données brutes
│   └── nyc-rolling-sales.csv
├── notebooks/                   # Notebooks reproductibles
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_models.ipynb
│   └── 03_finetuned_model.ipynb
├── models/                      # Modèles sérialisés
├── reports/figures/             # Rapport technique
├── utils/                       # Modules Python réutilisables
│   ├── data_prep.py             #   Nettoyage & détection outliers
│   ├── train.py                 #   Entraînement & feature engineering
│   ├── metrics.py               #   Évaluation des modèles
│   └── infer.py                 #   Scoring & analyse par déciles
├── requirements.txt
└── README.md
```