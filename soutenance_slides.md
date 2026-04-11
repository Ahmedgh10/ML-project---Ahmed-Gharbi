---
marp: true
title: Soutenance technique — Détection d’anomalies Smart System
paginate: true
---

# Détection d’anomalies Smart System (IoT)
## Classification multi-classes + explications + déploiement

- Candidat : (votre nom)
- Repo / App : Streamlit Cloud (démo)

---

# 1) Problématique & objectifs

**Contexte**
- Flux de télémétrie (timestamp, device_id/type, ressources, réseau, crypto, localisation)
- Attaques simulées : DoS, Injection, Spoofing + trafic Normal

**Objectifs**
- Détecter / classifier les anomalies (multi-classes)
- Évaluer la robustesse (plus réaliste qu’un split aléatoire)
- Fournir une explication actionnable pour chaque alerte

---

# 2) Données & préparation

**Source**
- Dataset : `smart_system_anomaly_dataset.csv`

**Schéma de features (exemples)**
- Numériques : CPU/mem/latence, bande passante, paquets, etc.
- Catégorielles : `device_type`, `protocol`, `region` …

**Pré-traitement**
- Imputation (médiane / mode)
- Standardisation des numériques
- One-hot encoding des catégorielles

---

# 3) Modélisation (≥ 3 modèles supervisés)

Pipeline scikit-learn : `ColumnTransformer` + modèle

**Modèles testés**
- Régression Logistique (baseline, `class_weight=balanced`)
- RandomForest (robuste aux non-linéarités)
- HistGradientBoosting (boosting rapide)

**Sorties**
- Label prédit + confiance (si `predict_proba`)

---

# 4) Stratégie d’évaluation (le “twist”)

Pourquoi c’est important : en prod, on prédit sur du **futur** ou des **devices jamais vus**.

**3 régimes de split**
- **Random** : stratifié (référence)
- **Time** : forward-in-time (réalisme)
- **Group** : unseen-device (split par `device_id`)

**Métriques**
- Macro-F1 (équilibrer les classes rares)
- Rapport de classification (précision/rappel par classe)

---

# 5) Explicabilité : “Nearest Normal Twin”

**Idée** : pour une ligne anormale, trouver la ligne **normale la plus proche** dans l’espace transformé.

**Procédure**
- On transforme les normales via le préprocesseur
- On entraîne un `NearestNeighbors` sur ces vecteurs
- Pour chaque anomalie : plus proche normal + **deltas** sur features clés

**Bénéfice**
- Explication simple : “ce qui a le plus changé vs un comportement normal similaire”

---

# 6) Application (Streamlit) — Prototype utilisable

**Flux utilisateur**
1. Upload CSV (avec ou sans `label`)
2. Table de prédictions (label + confiance)
3. Sélection d’une anomalie → explication (normal twin + deltas)
4. Download du CSV de sortie

**Bonus**
- Si `label` est présent : Macro-F1 + `classification_report` dans l’UI
- Graph : histogramme des labels prédits

---

# 7) Architecture & reproductibilité

**Offline**
- Entraînement : `train.py`
- Artefacts : `artifacts/model.joblib`, `normal_vectors.npy`, `normal_explain.csv`, `train_summary.json`

**Online**
- Inference : chargement artefacts → prédiction → explications
- App : `app.py` (Streamlit)

**Déploiement**
- GitHub + Streamlit Community Cloud
- `runtime.txt` pin Python 3.11 pour compatibilité scientifique

---

# 8) Résultats, limites & suites

**Résultats (à commenter pendant la démo)**
- Macro-F1 par régime (random vs time vs group)
- Exemples d’alertes expliquées (DoS / Injection / Spoofing)

**Limites**
- Modèle supervisé dépend des labels disponibles
- Généralisation : devices/protocoles/regions nouveaux

**Suites (court terme)**
- Sélection de seuil (rejeter faible confiance)
- Calibration des probabilités
- Analyse d’erreurs (confusions entre classes d’attaque)
