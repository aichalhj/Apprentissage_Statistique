#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC

from svm_source import *
from sklearn import svm
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from time import time
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.datasets import make_classification

scaler = StandardScaler()

import warnings
warnings.filterwarnings("ignore")

plt.style.use('ggplot')

#%%
#QUESTION 1
# Chargement des données

iris = datasets.load_iris()
X = iris.data
X = scaler.fit_transform(X)
y = iris.target
X = X[y != 0, :2]
y = y[y != 0]

# Tracer les points des classes 1 et 2
plt.figure(figsize=(8, 6))
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', label='Classe 1')
plt.scatter(X[y == 2, 0], X[y == 2, 1], color='orange', label='Classe 2')

# Personnalisation du graphique
plt.title('Classes 1 et 2 du dataset Iris ')
plt.xlabel('Variable 1')
plt.ylabel('Variable 2')
plt.legend()
plt.grid(True)
plt.show()

X, y = shuffle(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
parameters = {'kernel': ['linear'], 'C': list(np.logspace(-3, 3, 200))}
clf_linear = GridSearchCV(SVC(), parameters, cv=5)
clf_linear.fit(X_train, y_train)
y_pred = clf_linear.predict(X_test)

meilleur_C = clf_linear.best_params_['C']
print("Meilleure valeur de C : ", meilleur_C)

meilleur_score = clf_linear.best_score_
print("Meilleur score obtenu avec la meilleure valeur de C : ", meilleur_score)


# Calculer la matrice de confusion
cm = confusion_matrix(y_test, y_pred)
print("Matrice de confusion :\n", cm)

# Affichage de la matrice de confusion
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Classe 1', 'Classe 2'])
disp.plot(cmap=plt.cm.Blues)
plt.title('Matrice de Confusion')
plt.show()

print('Generalization score for linear kernel: %s, %s' %
      (clf_linear.score(X_train, y_train),
       clf_linear.score(X_test, y_test)))

def f_linear(xx):
    """Classifier: needed to avoid warning due to shape issues"""
    return clf_linear.predict(xx.reshape(1, -1))

frontiere(f_linear, X, y)
plt.title("Frontière de décision SVM linéaire")
plt.scatter([], [], color='blue', label='Classe 1')  
plt.scatter([], [], color='orange', label='Classe 2')  # Classe 2 en orange
plt.legend(loc='best') 
plt.show()
#%%
#QUESTION 2

Cs = list(np.logspace(-3, 3, 5))
gammas = 10. ** np.arange(1, 2)
degrees = np.r_[1, 2, 3]

parameters1 = {'kernel': ['poly'], 'C': Cs, 'gamma': gammas, 'degree': degrees}

clf_poly = GridSearchCV(SVC(), parameters1, n_jobs=-1)
clf_poly.fit(X_train, y_train)
y_pred_poly = clf_poly.predict(X_test)

print('Generalization score for polynomial kernel: %s, %s' %
      (clf_poly.score(X_train, y_train),
       clf_poly.score(X_test, y_test)))

def f_poly(xx):
    """Classifier: needed to avoid warning due to shape issues"""
    return clf_poly.predict(xx.reshape(1, -1))

frontiere(f_poly, X, y)
plt.title("Frontière de décision SVM polynomiale")
plt.scatter([], [], color='blue', label='Classe 1')  
plt.scatter([], [], color='orange', label='Classe 2')  # Classe 2 en orange
plt.legend(loc='best') 
plt.show()

#%%
#QUESTION 3

# Générer un jeu de données déséquilibré avec plus de bruit
X, y = make_classification(n_samples=1000,     # Nombre total d'exemples
                           n_features=2,       # Nombre de caractéristiques
                           n_informative=2,    # Nombre de caractéristiques informatives
                           n_redundant=0,      # Pas de caractéristiques redondantes
                           n_clusters_per_class=1,
                           weights=[0.9, 0.1], # 90% pour la classe 0, 10% pour la classe 1
                           flip_y=0.1,         # Introduire 10% de bruit dans les labels
                           class_sep=0.5,      # Réduire la séparation entre les classes
                           random_state=42)

# Ajouter du bruit gaussien aux caractéristiques
noise = np.random.normal(0, 1, X.shape)
X_noisy = X + noise * 0.5  # Multiplier par un facteur pour ajuster le niveau de bruit

# Visualiser les données désordonnées
plt.scatter(X_noisy[y == 0][:, 0], X_noisy[y == 0][:, 1], color='blue', label='Classe 0 (90%)')
plt.scatter(X_noisy[y == 1][:, 0], X_noisy[y == 1][:, 1], color='orange', label='Classe 1 (10%)')
plt.title('Jeu de données désordonné (avec bruit)')
plt.legend()
plt.show()

# Diviser les données en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Entraîner un SVM avec un noyau linéaire pour différentes valeurs de C
C_values = [1e-3, 1e-2, 1e-1, 1, 10, 100]
for C in C_values:
    svm_model = SVC(kernel='linear', C=C)
    svm_model.fit(X_train, y_train)
    y_pred = svm_model.predict(X_test)
    
    # Afficher la performance
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f'Pour C={C}:')
    print(f' - Accuracy: {acc}')
    print(f' - Matrice de confusion:\n{cm}\n')


# SVM avec class_weight='balanced'
svm_balanced = SVC(kernel='linear', C=1, class_weight='balanced')
svm_balanced.fit(X_train, y_train)
y_pred_balanced = svm_balanced.predict(X_test)

# Évaluer la performance
acc_balanced = accuracy_score(y_test, y_pred_balanced)
cm_balanced = confusion_matrix(y_test, y_pred_balanced)

print(f'Accuracy avec class_weight=\'balanced\': {acc_balanced}')
print(f'Matrice de confusion avec class_weight=\'balanced\':\n{cm_balanced}')
print(f'Pour C={C}:')

#%%
#QUESTION 4

# Download the data and unzip; then load it as numpy arrays
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4,
                              color=True, funneled=False, slice_=None,
                              download_if_missing=True)


# introspect the images arrays to find the shapes (for plotting)
images = lfw_people.images
n_samples, h, w, n_colors = images.shape

# the label to predict is the id of the person
target_names = lfw_people.target_names.tolist()

# Pick a pair to classify such as
names = ['Tony Blair', 'Colin Powell']


idx0 = (lfw_people.target == target_names.index(names[0]))
idx1 = (lfw_people.target == target_names.index(names[1]))
images = np.r_[images[idx0], images[idx1]]
n_samples = images.shape[0]
y = np.r_[np.zeros(np.sum(idx0)), np.ones(np.sum(idx1))].astype(int)

# plot a sample set of the data
plot_gallery(images, np.arange(12))
plt.show()

X = (np.mean(images, axis=3)).reshape(n_samples, -1)
# Scale features
X -= np.mean(X, axis=0)
X /= np.std(X, axis=0)

# split train test
X_train, X_test, y_train, y_test, images_train, images_test = \
   train_test_split(X, y, images, test_size=0.5, random_state=0)
X_train, X_test, y_train, y_test = \
   train_test_split(X, y, test_size=0.5, random_state=0)

# Split data into a half training and half test set
indices = np.random.permutation(X.shape[0])
train_idx, test_idx = indices[:X.shape[0] // 2], indices[X.shape[0] // 2:]
X_train, X_test = X[train_idx, :], X[test_idx, :]
y_train, y_test = y[train_idx], y[test_idx]
images_train, images_test = images[train_idx, :, :, :], images[test_idx, :, :, :]

print("--- Linear kernel ---")
print("Fitting the classifier to the training set")
t0 = time()

Cs = 10. ** np.arange(-5, 6)
scores = []
for C in Cs:
    clf = svm.SVC(kernel='linear', C=C)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    scores.append(accuracy_score(y_test, y_pred))

t1 = time()
print("done in %0.3fs" % (t1 - t0))

ind = np.argmax(scores)
print("Best C: {}".format(Cs[ind]))

plt.figure()
plt.plot(Cs, scores)
plt.xlabel("Parametres de regularisation C")
plt.ylabel("Scores d'apprentissage")
plt.xscale("log")
plt.tight_layout()
plt.show()
print("Best score: {}".format(np.max(scores)))

print("Predicting the people names on the testing set")
t0 = time()

# Qualitative evaluation of the predictions using matplotlib

prediction_titles = [title(y_pred[i], y_test[i], names)
                     for i in range(y_pred.shape[0])]

plot_gallery(images_test, prediction_titles)
plt.show()

plt.figure()
plt.imshow(np.reshape(clf.coef_, (h, w)))
plt.show()

#%%
#Question 5

def run_svm_cv(_X, _y):
    _indices = np.random.permutation(_X.shape[0])
    _train_idx, _test_idx = _indices[:_X.shape[0] // 2], _indices[_X.shape[0] // 2:]
    _X_train, _X_test = _X[_train_idx, :], _X[_test_idx, :]
    _y_train, _y_test = _y[_train_idx], _y[_test_idx]

    _parameters = {'kernel': ['linear'], 'C': list(np.logspace(-3, 3, 5))}
    _svr = SVC()
    _clf_linear = GridSearchCV(_svr, _parameters)
    _clf_linear.fit(_X_train, _y_train)

    print('Generalization score for linear kernel: %s, %s \n' %
          (_clf_linear.score(_X_train, _y_train), _clf_linear.score(_X_test, _y_test)))
    

n_features = X.shape[1]
# On rajoute des variables de nuisances
sigma = 1
noise = sigma * np.random.randn(n_samples, 300, )
X_noisy = np.concatenate((X, noise), axis=1)
X_noisy = X_noisy[np.random.permutation(X.shape[0])]

X_noisy_train, X_noisy_test, y_train, y_test = train_test_split(X_noisy, y, test_size=0.5)


# Fonction pour évaluer le modèle SVM
def evaluate_svm(_X_train, _X_test, _y_train, _y_test, C_values):
    test_scores = []

    for C in C_values:
        # Création du modèle SVM avec le noyau linéaire et un paramètre C donné
        svm = SVC(kernel='linear', C=C)
        svm.fit(_X_train, _y_train)

        # Évaluation sur les données de test
        score = svm.score(_X_test, _y_test)
        test_scores.append(score)

    return test_scores

# Paramètres de régularisation C à tester
C_values = np.logspace(-3, 3, 10)

# Evaluation sur les données sans bruit
scores_clean = evaluate_svm(X_train, X_test, y_train, y_test, C_values)

# Evaluation sur les données bruitées
scores_noisy = evaluate_svm(X_noisy_train, X_noisy_test, y_train, y_test, C_values)

# Tracer les résultats
plt.figure(figsize=(10, 6))
plt.plot(C_values, scores_clean, label='Score de test (Données Bruités)', marker='o')
plt.plot(C_values, scores_noisy, label='Score de test (Données non Bruités)', marker='o')
plt.xscale('log')
plt.xlabel('Paramètre de régularisation C (log scale)')
plt.ylabel('Score')
plt.title('Score de test en fonction de C pour les données bruitées et non bruitées')
plt.legend()
plt.grid(True)
plt.show()

run_svm_cv(X_noisy, y)

#%%
#QUESTION 6
n_components = 150  # jouer avec ce parametre
pca = PCA(n_components=n_components).fit(X_noisy)

X_noisy_pca = pca.transform(X_noisy)
run_svm_cv(X_noisy_pca, y)






