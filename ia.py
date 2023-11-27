import pandas as pd
from joblib import load

# Charger le modèle depuis le fichier
model = load('fraude.joblib')

# Charger le nouveau CSV dans un DataFrame (remplacez 'nouveau_fichier.csv' par le chemin réel de votre nouveau fichier CSV)
new_data = pd.read_csv('clean.csv', delimiter=',')

# Sélectionner les colonnes pertinentes (ajustez selon vos besoins)
new_features = new_data[['oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'amount']]

# Faire des prédictions sur les nouvelles données
predictions = model.predict(new_features)

# Ajouter les prédictions au DataFrame original
new_data['isFraudPrediction'] = predictions

# Filtrer les transactions prédites comme frauduleuses
fraudulent_transactions = new_data[new_data['isFraudPrediction'] == 1]

# Afficher les transactions frauduleuses
print("Transactions Frauduleuses :")
print(fraudulent_transactions)

# Afficher le total des transactions frauduleuses
total_fraudulent_transactions = len(fraudulent_transactions)
print(f"\nTotal des Transactions Frauduleuses : {total_fraudulent_transactions}")

# Écrire le DataFrame résultant dans un fichier CSV (remplacez 'resultats_fraude.csv' par le nom de fichier que vous souhaitez)
fraudulent_transactions.to_csv('resultats_fraude.csv', index=False)
