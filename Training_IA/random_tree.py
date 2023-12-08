import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Charger le CSV dans un DataFrame (remplacez 'votre_fichier.csv' par le chemin réel de votre fichier CSV)
df = pd.read_csv('Imputed.csv', delimiter=',')

# Sélectionner les colonnes pertinentes (ajustez selon vos besoins)
features = df[['oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'amount']]

# Sélectionner la colonne cible
target = df['isFraud']

# Diviser les données en ensemble d'entraînement et ensemble de test
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Initialiser le modèle de forêt aléatoire
random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Entraîner le modèle
random_forest_model.fit(X_train, y_train)

# Faire des prédictions sur l'ensemble de test
predictions = random_forest_model.predict(X_test)

# Calculer et afficher l'exactitude (accuracy) ainsi que d'autres métriques de classification
accuracy = accuracy_score(y_test, predictions)
conf_matrix = confusion_matrix(y_test, predictions)
classification_rep = classification_report(y_test, predictions)

print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{classification_rep}')
