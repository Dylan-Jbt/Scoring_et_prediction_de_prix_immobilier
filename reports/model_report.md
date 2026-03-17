# Model Report — Scoring & Prédiction de Prix Immobilier (Régression)

**Date** : 17 mars 2026  
**Auteur** : Data Science Project  
**Seed** : 1204  
**Fichier modèle final** : `models/finetuned.joblib`

---

## 1. Contexte et Objectif

### 1.1 — Problématique métier

Le projet vise à concevoir un modèle de machine learning produisant un **score de valorisation immobilière** pour détecter anomalies et opportunités d'investissement sur le marché new-yorkais.

Le modèle doit :
1. **Prédire un prix de vente théorique** d'un bien immobilier à partir de ses caractéristiques (localisation, type de bien, surface, année de construction, etc.).
2. **Comparer ce prix estimé au prix réel** de la transaction.
3. **Générer un score de valorisation** :
   - Score > 0 → bien en **surcote** (vendu au-dessus de l'estimation)
   - Score < 0 → bien en **décote** (opportunité d'investissement potentielle)

### 1.2 — Type de problème

- **Régression supervisée** : prédiction d'une variable continue (prix de vente en $).
- **Métrique principale** : R² (variance expliquée) + MAPE (erreur en % interprétable métier).

---

## 2. Données

### 2.1 — Source

- **Dataset** : NYC Rolling Sales (`data/nyc-rolling-sales.csv`)
- **Volume** : 84 548 transactions immobilières, 22 colonnes
- **Périmètre** : Ventes immobilières dans les 5 boroughs de New York City

### 2.2 — Variables

| Type | Nombre | Exemples |
|------|--------|----------|
| Catégorielles | 11 | `NEIGHBORHOOD`, `BUILDING CLASS CATEGORY`, `TAX CLASS AT PRESENT`, `ADDRESS`, `SALE DATE` |
| Numériques | 10 | `BOROUGH`, `BLOCK`, `LOT`, `ZIP CODE`, `RESIDENTIAL UNITS`, `COMMERCIAL UNITS`, `TOTAL UNITS`, `YEAR BUILT`, `TAX CLASS AT TIME OF SALE` |
| **Cible** | 1 | `SALE PRICE` |

### 2.3 — Qualité des données

| Critère | Valeur |
|---------|--------|
| Valeurs manquantes (features) | **0 %** (dataset complet) |
| Valeurs manquantes (cible) | **17,36 %** (1 736 / 10 000) |
| Doublons exacts | **0** |
| Doublons hors index | 30 (0,30 %) |
| Transactions à $0 | 1 154 (transferts intrafamiliaux / administratifs) |

### 2.4 — Split train/test

| Ensemble | Lignes (brut) | Lignes (après nettoyage NB02) | Lignes (après filtrage NB03) |
|----------|--------------|-------------------------------|------------------------------|
| Train | 10 000 (plafonné) | 7 110 | 7 002 |
| Test | 16 910 | 12 007 | 11 769 |

- Split 80/20, `random_state=1204`
- Jeu d'entraînement plafonné à 10 000 lignes

---

## 3. Analyse Exploratoire (Notebook 01)

### 3.1 — Variable cible : SALE PRICE

| Statistique | Valeur (prix > $0) |
|-------------|-------------------|
| Moyenne | $1 469 809 |
| Médiane | $630 000 |
| Écart-type | $6 451 817 |
| Min | $1 |
| Max | $268 124 175 |
| Skewness | **23,20** (très asymétrique) |
| Kurtosis | **715,98** |
| Ratio Moyenne/Médiane | **2,33** |

**Diagnostic** : Distribution extrêmement asymétrique à droite → **transformation `log1p` nécessaire** pour normaliser la cible avant modélisation.

### 3.2 — Corrélations avec SALE PRICE

| Variable | Corrélation | Force |
|----------|-------------|-------|
| TAX CLASS AT TIME OF SALE | +0,180 | Modérée |
| TOTAL UNITS | +0,140 | Faible |
| BOROUGH | -0,129 | Faible |
| RESIDENTIAL UNITS | +0,128 | Faible |
| BLOCK | -0,108 | Faible |

**Constat** : Aucune corrélation linéaire > 0,20 → les relations prix/features sont **non linéaires**, justifiant l'utilisation de modèles d'ensemble.

### 3.3 — Variables catégorielles (Test de Kruskal-Wallis)

**10/10 variables catégorielles significatives** (α = 0,05) :

| Variable | Statistique H | Interprétation |
|----------|--------------|----------------|
| NEIGHBORHOOD | 2 481 | Premier déterminant du prix |
| BUILDING CLASS AT TIME OF SALE | 1 499 | Type de bâtiment très discriminant |
| BUILDING CLASS AT PRESENT | 1 467 | Idem |
| BUILDING CLASS CATEGORY | ~1 400 | Granularité plus élevée |

---

## 4. Preprocessing

### 4.1 — Nettoyage

| Étape | Description | Impact |
|-------|-------------|--------|
| Suppression NaN/prix ≤ 0 | Retrait des cibles manquantes et prix nuls | −29 % du train (10 000 → 7 110) |
| Filtrage aberrants (NB03) | Exclusion des prix ≤ $1 000 | −108 lignes train, −238 test |
| Winsorization (P1–P99) | NB02 : [$10 — $12,9M] / NB03 : [$25 000 — $12,9M] | Réduction de l'impact des extrêmes |
| Transformation log1p | Appliquée à `SALE PRICE` | Skewness réduit, distribution quasi-symétrique |

### 4.2 — Pipeline de préprocessing

```
ColumnTransformer
├── Numériques (10 → 16 avec feature engineering)
│   ├── SimpleImputer(strategy='median')
│   └── StandardScaler()
└── Catégorielles (11 colonnes)
    ├── SimpleImputer(strategy='most_frequent')
    └── OneHotEncoder(handle_unknown='ignore', max_categories=30)
```

### 4.3 — Feature Engineering (Notebook 03)

| Feature créée | Formule | Justification |
|--------------|---------|---------------|
| `BUILDING_AGE` | 2026 − `YEAR BUILT` | Capture la vétusté du bien |
| `IS_NEW_BUILD` | 1 si âge < 20 ans | Premium des constructions récentes |
| `RATIO_GROSS_LAND` | Surface bâtie / Surface terrain | Densité de construction |
| `UNITS_RATIO_RES` | Unités résidentielles / Total unités | Différencie biens mixtes |
| `IS_RESIDENTIAL` | 1 si 0 unités commerciales | Segmente résidentiel pur |
| `GROSS_SF_x_BOROUGH` | Surface bâtie × Borough | Interaction surface × localisation |

---

## 5. Modèles Baseline (Notebook 02)

### 5.1 — Résultats

| Modèle | R² (train) | R² (test) | RMSE (test) | CV R² (5 folds) | Diagnostic |
|--------|-----------|-----------|-------------|-----------------|------------|
| Régression Linéaire | 0,2219 | 0,0772 | 1,7076 | 0,1021 ± 0,0456 | Stable, mais faible |
| XGBoost | 0,6400 | **-0,0158** | 1,7915 | 0,0917 ± 0,0172 | **Overfitting sévère** |
| **Random Forest** | 0,6763 | **0,1202** | 1,6673 | 0,1806 ± 0,0408 | Meilleur baseline |

### 5.2 — Importance des variables (baseline)

**XGBoost** : s'appuie sur des modalités catégorielles spécifiques (CONDOS ELEVATOR = 5 %, GROSS SQUARE FEET infrequent = 3 %).

**Random Forest** : privilégie les variables structurelles (`BLOCK` = 13 %, `LOT` = 11 %, `BOROUGH` = 10 %, `YEAR BUILT` = 6,5 %).

### 5.3 — Diagnostic

- Les performances sont **insuffisantes** (R² max = 0,12 en test).
- L'overfitting de XGBoost provient d'hyperparamètres par défaut trop expressifs.
- Le MAPE est inexploitable (> 60 000 %) à cause des transactions à très bas prix ($10–$100).
- **Causes identifiées** : pas de feature engineering, pas de filtrage des prix aberrants, hyperparamètres par défaut.

---

## 6. Modèles Finetuned (Notebook 03)

### 6.1 — Régression Régularisée (GridSearch + RandomizedSearch)

| Modèle | Meilleur α | l1_ratio | R² (CV) |
|--------|-----------|----------|---------|
| Ridge | 0,1 (Grid) / 4,76 (Random) | — | 0,3782 / 0,3787 |
| Lasso | 0,0001 | — | 0,3995 |
| **ElasticNet** | 0,000181 | 0,84 | **0,3999** |

**Gain vs baseline linéaire** : R² CV de 0,10 → 0,40 (+0,30 points), grâce au feature engineering et au filtrage des aberrants.

### 6.2 — XGBoost (RandomizedSearchCV + Optuna)

| Méthode | R² (CV) | Meilleurs hyperparamètres |
|---------|---------|--------------------------|
| RandomizedSearch (80 iter) | 0,6286 | `max_depth=9`, `lr=0.025`, `n_est=440`, `subsample=0.66`, `reg_lambda=5.95` |
| **Optuna (50 trials)** | **0,6303** | `max_depth=11`, `lr=0.017`, `n_est=456`, `subsample=0.91`, `reg_alpha=0.39`, `reg_lambda=5.37` |

### 6.3 — Random Forest (RandomizedSearchCV + Optuna)

| Méthode | R² (CV) | Meilleurs hyperparamètres |
|---------|---------|--------------------------|
| RandomizedSearch (80 iter) | 0,6256 | `max_depth=29`, `max_features=0.3`, `n_est=476`, `min_samples_split=3` |
| Optuna (50 trials) | 0,6255 | `max_depth=29`, `max_features=0.3`, `n_est=477`, `min_samples_split=3` |

**Observation** : Optuna apporte un gain marginal (+0,002 pour XGBoost, nul pour RF) — le RandomizedSearch à 80 itérations avait déjà quasi-convergé. Le plateau ~0,63 reflète les limites intrinsèques des données.

### 6.4 — Stacking

| Composant | Rôle |
|-----------|------|
| ElasticNet (finetuné) | Estimateur de base 1 (composante linéaire) |
| XGBoost (Optuna) | Estimateur de base 2 (interactions non linéaires) |
| Random Forest (Optuna) | Estimateur de base 3 (robustesse) |
| **Ridge (α=1)** | **Méta-apprenant** |

| Métrique | Valeur |
|----------|--------|
| R² OOF | **0,6363** |
| RMSE OOF | 0,6050 |
| R² Train | 0,8952 |

**Le stacking surpasse** les modèles individuels grâce à la complémentarité des estimateurs.

### 6.5 — Nested Cross-Validation (5 × 5)

| Modèle | R² train | R² test | RMSE test |
|--------|---------|---------|-----------|
| XGBoost | 0,8558 ± 0,053 | 0,6173 ± 0,016 | 0,6205 |
| Random Forest | 0,8632 ± 0,001 | 0,6183 ± 0,020 | 0,6197 |
| ElasticNet | 0,5193 ± 0,004 | 0,3999 ± 0,074 | 0,7759 |

Les estimations Nested CV confirment les performances CV standard avec une faible variance inter-folds.

---

## 7. Comparaison Globale (OOF)

| Modèle | RMSE | MAE | R² | MAPE |
|--------|------|-----|-----|------|
| Rég. Linéaire (baseline) | — | — | ~0,10 | — |
| XGBoost (baseline) | — | — | ~-0,02 | — |
| Random Forest (baseline) | — | — | ~0,18 | — |
| ElasticNet (finetuné) | — | — | ~0,40 | — |
| XGBoost (finetuné) | — | — | ~0,63 | — |
| Random Forest (finetuné) | — | — | ~0,63 | — |
| XGBoost (Optuna) | — | — | ~0,63 | — |
| Random Forest (Optuna) | — | — | ~0,63 | — |
| **Stacking** | **0,6050** | — | **0,6363** | — |

**Progression** : R² de 0,10 (baseline) → 0,64 (stacking finetuné) = **+0,54 points**.

---

## 8. Modèle Retenu : Stacking

### 8.1 — Évaluation finale sur le jeu de test

| Métrique | Valeur |
|----------|--------|
| **R² (test)** | **0,6073** |
| RMSE (log) | 0,6203 |
| MAE (log) | 0,3985 |
| MAPE | 67,76 % |
| Erreur médiane absolue | $150 019 |
| Erreur moyenne absolue | $491 447 |
| % prédictions à ±25 % | **50,5 %** |
| % prédictions à ±50 % | **76,0 %** |

### 8.2 — Performance par segment de prix

| Segment | N | R² | RMSE | MAPE |
|---------|---|-----|------|------|
| Bas (< $382K) | — | Plus faible | Plus élevé | Plus élevé |
| Standard ($382K – $1,1M) | — | **Meilleur** | **Plus bas** | **Plus bas** |
| Luxe (≥ $1,1M) | — | Intermédiaire | Intermédiaire | Intermédiaire |

Le modèle est le plus performant sur le **segment standard**, qui représente le cœur du marché résidentiel new-yorkais.

### 8.3 — Variables les plus importantes (consensus XGBoost / RF)

Variables communes au Top 10 des deux modèles :
- **`BOROUGH`** : l'arrondissement est un déterminant structurel du prix.
- **`TOTAL UNITS`** : le nombre total d'unités capture la taille/type du bien.

---

## 9. Artefacts produits

| Fichier | Description |
|---------|-------------|
| `models/baseline_lr.joblib` | Pipeline Régression Linéaire (baseline) |
| `models/baseline_xgb.joblib` | Pipeline XGBoost (baseline) |
| `models/baseline_rf.joblib` | Pipeline Random Forest (baseline) |
| `models/finetuned.joblib` | **Pipeline Stacking final** (production) |

---

## 10. Limites et Améliorations Possibles

### 10.1 — Limites identifiées

1. **Données manquantes** : les surfaces (`LAND SQUARE FEET`, `GROSS SQUARE FEET`) sont encodées en texte et contiennent des valeurs invalides — une part importante de l'information de surface est perdue.
2. **Variables absentes** : pas d'information sur l'état du bien, la vue, la proximité des transports, le nombre d'étages, la présence d'ascenseur, etc.
3. **Transactions atypiques** : malgré le filtrage, des ventes à prix anormalement bas persistent (transferts entre proches à $25 000 pour des biens estimés > $1M).
4. **Segment luxe** : la haute variabilité des biens de prestige rend la prédiction intrinsèquement difficile.

### 10.2 — Pistes d'amélioration

1. **Enrichissement des données** : joindre des données externes (proximité métro, écoles, criminalité, données Census).
2. **Modèles spécialisés par segment** : entraîner un modèle dédié pour les biens < $500K, un pour $500K–$2M, un pour > $2M.
3. **Target encoding** : remplacer le OneHotEncoding à haute cardinalité par un encodage cible pour `NEIGHBORHOOD` et `ADDRESS`.
4. **Traitement avancé des surfaces** : parser et convertir systématiquement `LAND SQUARE FEET` et `GROSS SQUARE FEET` en numériques avant le split.
5. **Modèles supplémentaires** : LightGBM, CatBoost (optimisé pour les catégorielles).

---

## 11. Reproductibilité

| Paramètre | Valeur |
|-----------|--------|
| Python | 3.13 |
| Seed global | 1204 |
| Split | `train_test_split(test_size=0.2, random_state=1204)` |
| CV | `KFold(n_splits=5, shuffle=True, random_state=1204)` |
| Max train | 10 000 lignes |
| Dépendances | `requirements.txt` |

### Stack technique

- **Data** : pandas, numpy, scipy
- **ML** : scikit-learn, xgboost, optuna
- **Visualisation** : matplotlib, seaborn, plotly
- **Persistance** : joblib

---

## 12. Conclusion

Le projet démontre une démarche data science complète, de l'exploration à la mise en production :

- L'**analyse exploratoire** (NB01) a révélé une distribution extrêmement asymétrique, des transactions aberrantes et des relations non linéaires.
- Les **baselines** (NB02) ont confirmé les limites des modèles simples (R² = 0,12 max) et l'overfitting de XGBoost par défaut.
- Le **finetuning** (NB03) a transformé les performances via trois leviers : filtrage des aberrants, feature engineering (6 variables), et optimisation des hyperparamètres (RandomSearch, Optuna, Stacking).

Le modèle final (**Stacking**, R² = 0,61) prédit correctement l'ordre de grandeur du prix pour **76 % des biens** (à ±50 %) et peut servir de base à un scoring de valorisation immobilière pour identifier les opportunités d'investissement.