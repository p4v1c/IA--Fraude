import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Charger le CSV dans un DataFrame (remplacez 'votre_fichier.csv' par le chemin réel de votre fichier CSV)
df = pd.read_csv('Imputed.csv', delimiter=',')

# Sélectionner les colonnes pertinentes (ajustez selon vos besoins)
features = df[['oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'amount']]

# Sélectionner la colonne cible
target = df['isFraud']

# Diviser les données en ensemble d'entraînement et ensemble de test
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Initialiser le modèle de régression linéaire
linear_model = LinearRegression()

# Entraîner le modèle
linear_model.fit(X_train, y_train)

# Faire des prédictions sur l'ensemble de test
predictions = linear_model.predict(X_test)

# Calculer et afficher l'erreur quadratique moyenne
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')
