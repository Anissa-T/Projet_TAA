"""
Ce module traite des données bilingues (anglais-français) de l'ensemble de données EMEA.
Les principales étapes du traitement des données comprennent le chargement, la division et la sauvegarde des données.

Bibliothèques utilisées :
- pandas : Pour la manipulation et l'analyse des données.
- train_test_split de sklearn.model_selection : Pour diviser les données en ensembles d'entraînement et de test.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

# Lire les fichiers contenant les données en anglais et en français
file_path_en = '../data/Emea/EMEA.en-fr.en'
file_path_fr = '../data/Emea/EMEA.en-fr.fr'

# Charger les données ligne par ligne
with open(file_path_en, 'r', encoding='utf-8') as f_en:
    data_en = f_en.readlines()

with open(file_path_fr, 'r', encoding='utf-8') as f_fr:
    data_fr = f_fr.readlines()

# Vérifier si les deux fichiers ont le même nombre de lignes
assert len(data_en) == len(data_fr), "Les fichiers anglais et français ne correspondent pas en taille."

# Créer des DataFrames
data = pd.DataFrame({'en': data_en, 'fr': data_fr})

# Sélectionner les premières 11500 phrases pour respecter les tailles demandées (10k + 1k + 500)
data = data.iloc[:11500]

# Diviser les données en ensembles d'apprentissage, de validation et de test
X = data['en']
y = data['fr']

# Diviser d'abord pour obtenir 10k et le reste (1.5k)
X_train, X_rest, y_train, y_rest = train_test_split(X, y, train_size=10000, random_state=42, shuffle=True)

# Diviser le reste pour obtenir 1k de validation et 500 de test
X_val, X_test, y_val, y_test = train_test_split(X_rest, y_rest, train_size=1000, random_state=42, shuffle=True)

# Affichage des résultats
print("Taille de X_train:", len(X_train))  # Doit être 10,000
print("Taille de X_val:", len(X_val))      # Doit être 1,000
print("Taille de X_test:", len(X_test))    # Doit être 500
print("Taille de y_train:", len(y_train))
print("Taille de y_val:", len(y_val))
print("Taille de y_test:", len(y_test))

# Enregistrer les ensembles de données dans des fichiers séparés
X_train.to_csv('../data/Emea/EMEA_train_10k.en', index=False, header=False)
y_train.to_csv('../data/Emea/EMEA_train_10k.fr', index=False, header=False)
X_val.to_csv('../data/Emea/EMEA_dev_1k.en', index=False, header=False)
y_val.to_csv('../data/Emea/EMEA_dev_1k.fr', index=False, header=False)
X_test.to_csv('../data/Emea/EMEA_test_500.en', index=False, header=False)
y_test.to_csv('../data/Emea/EMEA_test_500.fr', index=False, header=False)
