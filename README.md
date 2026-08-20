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

```
Données brutes (CSV)
        │
        ▼
Nettoyage & split stratifié (80/20)
        │
        ▼
Tokenisation (AutoTokenizer, sous-mots WordPiece)
        │
        ├──► Fine-tuning DistilBERT  ──┐
        │                              ├──► Comparaison & conclusion
        └──► Fine-tuning BERT-base  ──┘
```

Les deux modèles reposent sur le même principe de **transfer learning** : on réutilise un modèle pré-entraîné sur un corpus massif, et on fine-tune l'ensemble du réseau (poids + nouvelle tête de classification à 2 classes) sur la tâche spam/ham, via l'API `Trainer` de HuggingFace. Seul le checkpoint pré-entraîné change entre les deux runs, pour une comparaison strictement équitable.

---

## 📊 Résultats

Deux modèles [HuggingFace Transformers](https://huggingface.co/) ont été fine-tunés et comparés sur un jeu de test stratifié (1 115 SMS, dont 149 spam) :

| Modèle | Paramètres | F1-score (spam) | Precision | Recall | Accuracy |
|---|---|---|---|---|---|
| [`distilbert-base-uncased`](https://huggingface.co/distilbert/distilbert-base-uncased) | ~66M | **0.973** | 0.99 | 0.96 | 0.99 |
| [`bert-base-uncased`](https://huggingface.co/bert-base-uncased) | ~110M | **0.969** | 0.99 | 0.95 | 0.99 |

**À retenir** : le modèle léger (DistilBERT) égale, voire dépasse légèrement, le modèle plus lourd (BERT-base) sur cette tâche — les SMS étant courts, la capacité supplémentaire de BERT-base n'apporte pas de gain mesurable. **DistilBERT est donc le meilleur compromis** : performance équivalente pour ~40% de coût de calcul en moins (entraînement + inférence), un critère important pour un traitement en temps réel à grande échelle.

<details>
<summary>Détail complet (classification report)</summary>

```
DistilBERT
              precision    recall  f1-score   support
         ham       0.99      1.00      1.00       966
        spam       0.99      0.96      0.97       149

BERT-base
              precision    recall  f1-score   support
         ham       0.99      1.00      1.00       966
        spam       0.99      0.95      0.97       149
```
> 💡 Un GPU accélère significativement les sections 3 et 4 (fine-tuning), mais n'est pas indispensable.

## 📁 Structure du projet

```
.
├── 01-AT_T_spam_detector.ipynb   # Notebook principal (livrable)
├── data/
│   └── spam.csv                  # Dataset source
├── requirements.txt
└── README.md
```

## 🧩 Pistes d'amélioration

- [ ] Ajuster le seuil de décision selon le compromis precision/recall souhaité
- [ ] Étudier les faux négatifs pour identifier des patterns de spam non détectés
- [ ] Valider la robustesse sur des messages plus longs ou multilingues
- [ ] Mise en production (API FastAPI, conteneurisation Docker, monitoring)

