# Le cumule de tous les paramètres

import pandas as pd

# Chargement des données
data = pd.read_csv("data/hr_hiring_dataset.csv")

# Encodage des variables catégorielles (drop_first=True = on saut la première ligne)
data_encoded = pd.get_dummies(data.drop("Hiring Decision", axis=1), drop_first=True)

print(data)