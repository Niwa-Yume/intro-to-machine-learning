import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

data = pd.read_csv("data/hr_hiring_dataset.csv")

# Encodage des variables catégorielles
data_encoded = pd.get_dummies(data.drop("Hiring Decision", axis=1), drop_first=True)

# Séparation des variables d'entrée et de sortie
data_target = data["Hiring Decision"].map({'Hired': 1, 'Not Hired': 0})
x = data_encoded
y = data_target

# Séparation des données en ensembles d'entraînement et de test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=500)

# Normalisation des données pour améliorer les performances du modèle
scaler = StandardScaler()
# Appliquer la normalisation aux données d'entraînement
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#utiliser kera pour faire un réseau de neurone


#préparation de l'entrainment
model = Sequential()
model.add(Dense(12, input_dim=x_train.shape[1], activation='relu'))
model.add(Dense(8, activation='relu')) # Relu sert
model.add(Dense(1, activation='sigmoid')) #sigmoid a pour but de
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

#entrainement en se basant sur le jeu
#epochs sert à
#batch_size c'est l
#verbose sert comme un print
model.fit(x_train, y_train, epochs=50, batch_size=10, verbose=1)

y_pred = (model.predict(x_test) > 0.5).astype(int).flatten()

accuracy = accuracy_score(y_test, y_pred)

print(accuracy*100)


