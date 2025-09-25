import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

data = pd.read_csv("data/hr_hiring_dataset.csv")

# Encodage des variables catégorielles
data_encoded = pd.get_dummies(data.drop("Hiring Decision", axis=1), drop_first=True)

# Séparation des variables d'entrée et de sortie
data_target = data["Hiring Decision"].map({'Hired': 1, 'Not Hired': 0})
x = data_encoded
y = data_target

# Séparation des données en ensembles d'entraînement et de test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=500)



# Initialisation des variables pour suivre le meilleur score
best_accuracy = 0
best_params = {}

# Définition des plages de valeurs à tester
# Il est plus efficace de tester des plages raisonnables
max_depth_range = range(1, 10) # Teste des profondeurs de 1 à 500
min_samples_leaf_range = range(1, 10) # Teste de 1 à 500

print("Début de la recherche des meilleurs accuracy")

# Boucle sur toutes les profondeurs max à tester
for depth in max_depth_range:
    # Boucle sur toutes les tailles de feuille min à tester
    for leaf in min_samples_leaf_range:
        # 1. Création du classifieur avec les paramètres de la boucle actuelle
        clf = DecisionTreeClassifier(
            max_depth=depth,
            min_samples_leaf=leaf,
        )

        # 2. Entraînement du modèle
        clf.fit(x_train, y_train)

        # 3. Prédiction sur les données de test
        y_pred = clf.predict(x_test)

        # 4. Calcul de l'accuracy
        current_accuracy = accuracy_score(y_test, y_pred)

        # 5. Vérification si ce modèle est le meilleur jusqu'à présent
        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            best_params = {'max_depth': depth, 'min_samples_leaf': leaf}
            print(f"Nouveau meilleur score : {best_accuracy*100:.2f}% avec max_depth={depth}, min_samples_leaf={leaf}")

# --- Affichage des résultats finaux ---
print("\nRecherche terminée.")
print("--------------------------------------------------")
print(f"Meilleurs paramètres trouvés : {best_params}")
print(f"Meilleure accuracy sur l'ensemble de test : {best_accuracy * 100:.2f}%")


#affichage de l'arbe
plt.figure(figsize=(20,10))
plot_tree(clf, filled=True, feature_names=x.columns,
          class_names=['Not Hired', 'Hired'], rounded=True, fontsize=8)
plt.show()