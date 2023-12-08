import pandas as pd

# Charger le CSV dans un DataFrame (remplacez 'votre_fichier.csv' par le chemin réel de votre fichier CSV)
df = pd.read_csv('delete.csv', delimiter=',')

# Supprimer les lignes où à la fois 'oldbalanceOrg' et 'newbalanceOrig' sont égales à zéro
df = df.loc[(df['oldbalanceOrg'] != 0) | (df['newbalanceOrig'] != 0) ]
df = df.loc[(df['oldbalanceDest'] != 0) | (df['newbalanceDest'] != 0)]

# Afficher le DataFrame après la suppression
print("Après la suppression :\n", df)

# Écrire le DataFrame nettoyé dans un nouveau fichier CSV (remplacez 'nouveau_fichier.csv' par le nom souhaité)
df.to_csv('delete.csv', index=False, sep=',')

print("Le DataFrame nettoyé a été écrit dans 'delete.csv'")
