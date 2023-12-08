import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# Charger le CSV dans un DataFrame (remplacez 'votre_fichier.csv' par le chemin réel de votre fichier CSV)
df = pd.read_csv('Imputed.csv', delimiter=',')

# Supprimer les colonnes non numériques
df_numeric = df.select_dtypes(include=['float64', 'int64'])

# Convertir les valeurs numériques en binaire pour l'application de l'algorithme Apriori
df_binary = df_numeric.applymap(lambda x: 1 if x > 0 else 0)

# Appliquer l'algorithme Apriori
frequent_itemsets = apriori(df_binary, min_support=0.1, use_colnames=True)

# Générer les règles d'association
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

# Afficher les règles d'association
print(rules)
