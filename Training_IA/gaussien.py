import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score

# Charger le CSV dans un DataFrame (remplacez 'votre_fichier.csv' par le chemin réel de votre fichier CSV)
df = pd.read_csv('Imputed.csv', delimiter=',')

# Sélectionner les colonnes pertinentes (ajustez selon vos besoins)
features = df[['oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'amount']]

# Sélectionner la colonne cible
target = df['isFraud']

# Diviser les données en ensemble d'entraînement et ensemble de test
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Prétraitement des données
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_scaled)

# Initialiser le modèle de mélange gaussien
n_components = 2  # Vous pouvez ajuster le nombre de composants selon vos besoins
gmm_model = GaussianMixture(n_components=n_components, covariance_type='full', reg_covar=1e-3, random_state=42)

# Entraîner le modèle sur les données prétraitées
gmm_model.fit(X_train_pca)

# Prédire les clusters sur l'ensemble de test
X_test_scaled = scaler.transform(X_test)
X_test_pca = pca.transform(X_test_scaled)
predictions = gmm_model.predict(X_test_pca)

# Silhouette Score (métrique de la qualité du clustering)
silhouette = silhouette_score(X_test_pca, predictions)
print(f'Silhouette Score: {silhouette}')

# Ajustez cette partie en fonction de vos besoins spécifiques
# Par exemple, vous pouvez utiliser adjusted_rand_score pour mesurer l'ajustement aux vraies étiquettes si disponibles
true_labels = y_test  # Vous devrez remplacer ceci par vos vraies étiquettes si disponibles
rand_score = adjusted_rand_score(true_labels, predictions)
print(f'Adjusted Rand Score: {rand_score}')
