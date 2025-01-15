import pandas as pd
import ast

# Lire le fichier CSV
df = pd.read_csv('data.csv', sep=';')

# Fonction pour convertir les chaînes en listes
def process_dam(value):
    try:
        # Si la valeur commence par un crochet, utiliser ast.literal_eval
        if value.startswith('['):
            return ast.literal_eval(value)
        # Si la valeur contient des virgules, diviser par les virgules
        elif ',' in value:
            return [v.strip() for v in value.split(',')]
        else:
            # Sinon, encapsuler la valeur dans une liste
            return [value]
    except Exception as e:
        print(f"Erreur lors de l'évaluation de la valeur : {value}. Erreur : {e}")
        return [value]

# Appliquer la fonction à la colonne 'Dam'
df['Dam'] = df['Dam'].apply(process_dam)

# Afficher la liste corrigée
print(df['Dam'].to_list())
