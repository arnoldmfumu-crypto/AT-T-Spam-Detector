# 📵 SPAM Detector — AT&T

Projet réalisé dans le cadre de la formation Data Scientist (module IA, 150h).
Objectif : détecter automatiquement les SMS de type *spam* pour le compte d'AT&T, à partir du seul contenu textuel du message.

## 🎯 Contexte

AT&T souhaite automatiser la détection des SMS indésirables (spam) afin de protéger ses utilisateurs, la détection étant aujourd'hui faite manuellement.

**Livrable demandé** : un notebook qui exécute le preprocessing et entraîne un ou plusieurs modèles de deep learning capables de prédire la nature *spam* ou *ham* d'un SMS, avec des performances clairement énoncées.

> Ce dépôt ne couvre **pas** la mise en production (API, conteneurisation, monitoring) — le périmètre est volontairement limité au notebook de modélisation.

## 🗂️ Données

- `data/spam.csv` — dataset public [UCI SMS Spam Collection], 5 572 SMS étiquetés `ham` (légitime) ou `spam`.
- Distribution des classes : ~86% ham / ~14% spam (dataset déséquilibré).

## 🧠 Approche

Deux modèles de **deep learning HuggingFace**, tous deux basés sur le *transfer learning* mais de tailles différentes, sont construits et comparés — le même code de fine-tuning (`Trainer` API) est réutilisé pour les deux, seul le checkpoint change :

| Modèle | Description | Paramètres | Cours mobilisés |
|---|---|---|---|
| **Modèle 1** | Fine-tuning de `distilbert-base-uncased` (léger, rapide) | ~66M | `Understand_Transfer_Learning`, `Tokenization`, `Transformers_what_can_they_do`, `Behind_the_pipeline (PyTorch)`, `Fine_tuning_transfromers` |
| **Modèle 2** | Fine-tuning de `bert-base-uncased` (plus grand, même checkpoint que le cours) | ~110M | `Fine_tuning_transfromers` |

Le F1-score sur la classe `spam` est utilisé comme métrique principale plutôt que la seule accuracy, pour tenir compte du déséquilibre des classes (~86% ham / 14% spam).

## 📊 Résultats

| Modèle | F1-score (spam) | Precision (spam) | Recall (spam) | Accuracy |
|---|---|---|---|---|
| Modèle 1 — DistilBERT fine-tuné | **0.9728** | 0.99 | 0.96 | 0.99 |
| Modèle 2 — BERT-base fine-tuné | **0.9693** | 0.99 | 0.95 | 0.99 |

*(Jeu de test stratifié, 1115 SMS dont 149 spam)*

Les deux modèles atteignent un excellent niveau de performance, très proche l'un de l'autre — DistilBERT fait même légèrement mieux que BERT-base ici (meilleur recall sur la classe spam). Sur cette tâche (classifier des SMS courts), la capacité supplémentaire de BERT-base (~1.7x plus de paramètres) ne se traduit pas par un gain de performance : le vocabulaire "spam" (mots-clés promotionnels, majuscules, numéros, liens) est déjà bien capté par le modèle léger.

**Recommandation** : pour ce cas d'usage, **DistilBERT est le meilleur choix** — performance égale ou supérieure, pour un coût d'entraînement et d'inférence ~40% inférieur. C'est un point important pour AT&T si le modèle doit tourner en temps réel sur un grand volume de SMS entrants.

## 🚀 Utilisation

```bash
# 1. Créer un environnement virtuel (recommandé)
python -m venv venv
source venv/bin/activate   # ou venv\Scripts\activate sous Windows

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Lancer Jupyter et ouvrir le notebook
jupyter notebook 01-AT_T_spam_detector.ipynb
```

Le notebook s'exécute de haut en bas :
1. Chargement & exploration des données
2. Tokenisation pour les transformers (nécessite un accès internet au Hub HuggingFace)
3. Modèle 1 : DistilBERT fine-tuné (nécessite un accès internet)
4. Modèle 2 : BERT-base fine-tuné (nécessite un accès internet)
5. Comparaison des performances & conclusion — **résultats déjà obtenus, voir ci-dessus**

## 📁 Structure du projet

```
.
├── 01-AT_T_spam_detector.ipynb   # Notebook principal (livrable)
├── data/
│   └── spam.csv                  # Dataset source
├── requirements.txt
└── README.md
```

## ⚠️ Note sur `requirements.txt`

`scikit-learn` reste utile (split train/test, métriques), mais `TfidfVectorizer` n'est plus utilisé : les deux modèles sont désormais des transformers HuggingFace. Vous pouvez retirer cette dépendance si vous voulez alléger l'environnement, elle ne gêne pas si vous la gardez.

## 🧩 Pistes d'amélioration (hors périmètre de ce livrable)

- Ajuster le seuil de décision selon le compromis precision/recall souhaité par AT&T.
- Valider la robustesse sur des messages plus longs ou multilingues.
- Mise en production (API, conteneurisation, monitoring) — non traitée ici à la demande du commanditaire.

## ❓ Question ouverte pour vous

L'évaluation des modèles (classification report, matrice de confusion, F1-score) s'appuie sur `scikit-learn`, une bibliothèque standard non couverte par les cours qui m'ont été transmis. Si vous disposez d'un cours dédié à l'évaluation de modèles que vous souhaitez que je suive à la place (méthodologie, métriques spécifiques, seuils...), merci de me le transmettre et j'adapterai le notebook en conséquence.
