import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Charger le CSV dans un DataFrame (remplacez 'votre_fichier.csv' par le chemin réel de votre fichier CSV)
df = pd.read_csv('Imputed.csv', delimiter=',')

# Sélectionner les colonnes pertinentes (ajustez selon vos besoins)
features = df[['oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'amount']]

# Diviser les données en ensemble d'entraînement et ensemble de test
X_train, X_test = train_test_split(features, test_size=0.2, random_state=42)

# Initialiser le modèle KMeans pour la régression
kmeans_model = KMeans(n_clusters=2)  # Ajustez le nombre de clusters selon vos besoins

# Entraîner le modèle
kmeans_model.fit(X_train)

# Faire des prédictions sur l'ensemble de test
predictions = kmeans_model.predict(X_test)

# Calculer et afficher le score de silhouette
silhouette = silhouette_score(X_test, predictions)
print(f'Silhouette Score: {silhouette}')
