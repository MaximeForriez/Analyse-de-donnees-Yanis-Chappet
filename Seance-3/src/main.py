#coding:utf8

#coding:utf8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Source des données : https://www.data.gouv.fr/datasets/election-presidentielle-des-10-et-24-avril-2022-resultats-definitifs-du-1er-tour/
with open("./data/resultats-elections-presidentielles-2022-1er-tour.csv","r") as fichier:
    contenu = pd.read_csv(fichier)


# 1. Chargement des données

print("Question1")
# Lecture du fichier CSV (même source que la séance 2)
with open("./data/resultats-elections-presidentielles-2022-1er-tour.csv", "r", encoding="utf-8") as f:
    contenu = pd.read_csv(f)



# 2. Sélection des colonnes quantitatives

print("Question2")
colonnes_quanti = contenu.select_dtypes(include=[np.number]).columns
print("Colonnes quantitatives détectées :", list(colonnes_quanti), "\n")


# 3. Calcul des paramètres statistiques

print("Question3")
stats = pd.DataFrame(index=colonnes_quanti)

stats["Moyenne"] = contenu[colonnes_quanti].mean().round(2)
stats["Médiane"] = contenu[colonnes_quanti].median().round(2)
stats["Mode"] = contenu[colonnes_quanti].mode().iloc[0].round(2)
stats["Écart-type"] = contenu[colonnes_quanti].std().round(2)
stats["Écart absolu à la moyenne"] = (contenu[colonnes_quanti] - contenu[colonnes_quanti].mean()).abs().mean().round(2)
stats["Étendue"] = (contenu[colonnes_quanti].max() - contenu[colonnes_quanti].min()).round(2)
stats["Q1 (0.25)"] = contenu[colonnes_quanti].quantile(0.25).round(2)
stats["Q3 (0.75)"] = contenu[colonnes_quanti].quantile(0.75).round(2)
stats["IQR (Q3 - Q1)"] = (stats["Q3 (0.75)"] - stats["Q1 (0.25)"]).round(2)
stats["D1 (0.1)"] = contenu[colonnes_quanti].quantile(0.1).round(2)
stats["D9 (0.9)"] = contenu[colonnes_quanti].quantile(0.9).round(2)
stats["IDR (D9 - D1)"] = (stats["D9 (0.9)"] - stats["D1 (0.1)"]).round(2)

print("Paramètres statistiques :\n", stats, "\n")


# 4. Visualisations - Boîtes à moustaches

print("Question4")
import os
os.makedirs('./img', exist_ok=True)

for col in colonnes_quanti:
    plt.figure()
    plt.boxplot(contenu[col].dropna(), vert=True)
    plt.title(f"Boîte de dispersion - {col}")
    plt.ylabel(col)
    plt.tight_layout()
    plt.savefig(f"./img/boxplot_{col}.png")
    plt.close()

print("Boîtes à moustaches enregistrées dans le dossier ./img\n")


# 5. Catégorisation - fichier island-index.csv

print("Question5")
islands = pd.read_csv("./data/island-index.csv")
surfaces = islands["Surface (km²)"]

# Définition des classes et des étiquettes
bins = [0, 10, 25, 50, 100, 2500, 5000, 10000, np.inf]
labels = ["]0,10]", "]10,25]", "]25,50]", "]50,100]", "]100,2500]", "]2500,5000]", "]5000,10000]", "]10000,np.inf["]

islands["Catégorie_surface"] = pd.cut(surfaces, bins=bins, labels=labels, right=True)
print("Nombre d’îles par catégorie de surface :")
print(islands["Catégorie_surface"].value_counts().sort_index())
