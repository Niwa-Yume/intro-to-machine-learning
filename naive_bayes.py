# Le cumule de tous les paramètres

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

# Chargement des données
data = pd.read_csv("data/hr_hiring_dataset.csv")

# Encodage des variables catégorielles (drop_first=True = on saut la première ligne)
data_encoded = pd.get_dummies(data.drop("Hiring Decision", axis=1), drop_first=True)

# Séparation des variables d'entrée et de sortie
data_target = data["Hiring Decision"].map({'Hired': 1, 'Not Hired': 0})


x = data_encoded
y = data_target

# Séparation des données en ensembles d'entraînement et de test pour l'évaluation du modèle et l'optimisation des hyperparamètres
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=12345)

# Création du modèle Naive Bayes (Création du cerveau ia)
gnb = GaussianNB()

# Entraînement du modèle avec les données d'entraînement (Remplissage du cerveau)
gnb.fit(x_train, y_train)

#tentative de prédiction sur les données de test
y_pred = gnb.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)

print(accuracy*100)
