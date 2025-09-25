import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("data/hr_hiring_dataset.csv")
# Encodage des variables catégorielles (drop_first=True = on saut la première ligne)
data_encoded = pd.get_dummies(data.drop("Hiring Decision", axis=1), drop_first=True)

# Séparation des variables d'entrée et de sortie
data_target = data["Hiring Decision"].map({'Hired': 1, 'Not Hired': 0})
x = data_encoded
y = data_target

# Séparation des données en ensembles d'entraînement et de test pour l'évaluation du modèle et l'optimisation des hyperparamètres
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=1)

# Normalisation des données pour améliorer les performances du modèle
scaler = StandardScaler()

# Appliquer la normalisation aux données d'entraînement
x_train = scaler.fit_transform(x_train)
# Appliquer la même normalisation aux données de test
x_test = scaler.transform(x_test)

# Création du modèle KNN (Création du cerveau ia) qui classifie en se basant sur les X voisins les plus proches
knn = KNeighborsClassifier(n_neighbors=79) #80.9 pour 100
# Entraînement du modèle avec les données d'entraînement (Remplissage du cerveau)
knn.fit(x_train, y_train)

#tentative de prédiction sur les données de test
y_pred = knn.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)

print(accuracy*100)