# Apprentissage Statistique

Ce dépôt contient le code et l’analyse pour un TP portant sur les Support Vector Machines (SVM) dans le cadre du Master 2 Statistiques et Sciences des Données. L'objectif principal est d'explorer les capacités de classification des SVM en utilisant différents noyaux et hyperparamètres, tout en évaluant leurs performances en généralisation.



Les principales techniques explorées dans ce TP incluent :

- **SVM avec noyau linéaire et polynomial** : analyse des performances avec différents noyaux.
- **Optimisation des hyperparamètres** : utilisation de la validation croisée pour affiner les paramètres.
- **Gestion du déséquilibre des classes et de la régularisation (paramètre C)**.
- **Réduction de la dimensionnalité** : application de l'**ACP** (Analyse en Composantes Principales) pour simplifier l'espace de travail.

## Organisation du dépôt

Le dépôt est organisé de la manière suivante :

### Scripts Python
- **`svm_script.py`** : script à compléter fourni pour le TP.
- **`svm_gui.py`** : script permettant de lancer l'interface graphique du SVM.
- **`svm_source.py`** : script regroupant les fonctions nécessaires à la réalisation du TP.
- **`TP_SVM_script.py`** : script final, complété à partir du fichier `svm_script.py`.

### Documents PDF
- **`TP_ML_SVM.pdf`** : énoncé officiel du TP.
- **`TP_SVM_LAHJIOUJ_Aicha.pdf`** : version finale du TP rendue.

### Fichiers LaTeX
- **`TP3_SVM_LAHJIOUJ.tex`** : fichier source LaTeX du rendu du TP.

## Installation

Pour exécuter les scripts, les packages Python suivants sont requis :

- **Numpy**
- **Pandas**
- **Matplotlib**
- **Scikit-learn**
