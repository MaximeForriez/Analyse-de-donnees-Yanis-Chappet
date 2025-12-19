# coding: utf8

import pandas as pd
import matplotlib.pyplot as plt
import os


# Chargement des données


# Source des données :
# https://www.data.gouv.fr/datasets/election-presidentielle-des-10-et-24-avril-2022-resultats-definitifs-du-1er-tour/
contenu = pd.read_csv("./data/resultats-elections-presidentielles-2022-1er-tour.csv")


# Question 1 : colonnes

print(contenu.columns)


# Question 2 : dimensions

nb_lignes = len(contenu)
nb_colonnes = len(contenu.columns)
print(f"Nombre de lignes : {nb_lignes}")
print(f"Nombre de colonnes : {nb_colonnes}")


# Question 3 : types

print("Types de variables")
contenu.info()

print("Liste des types")
print(contenu.dtypes)


# Question 4 : noms colonnes

print("Nom des colonnes :")
print(contenu.columns)


# Question 5 : inscrits

print(
    "Nombre total d'inscrits aux élections présidentielles de 2022 :",
    contenu["Inscrits"].sum()
)


# Question 6 : sommes quantitatives

sommes = []
for colonne in contenu.columns:
    if contenu[colonne].dtype in ["int64", "float64"]:
        sommes.append(contenu[colonne].sum())

print("Sommes des colonnes quantitatives :", sommes)


# Question 7 : diagrammes en barres


os.makedirs("graphiques", exist_ok=True)

for _, row in contenu.iterrows():
    departement = row["Libellé du département"]
    inscrits = row["Inscrits"]
    votants = row["Votants"]

    categories = ["Inscrits", "Votants"]
    valeurs = [inscrits, votants]

    plt.figure(figsize=(8, 6))
    plt.bar(categories, valeurs)
    plt.title(f"Inscrits et Votants - {departement}")
    plt.ylabel("Nombre de personnes")

    nom_propre = departement.replace("/", "_")
    plt.savefig(f"graphiques/{nom_propre}.png", bbox_inches="tight")
    plt.close()

print("Diagrammes en barres créés avec succès")


# Question 8 : diagrammes circulaires


os.makedirs("graphiques_circulaires", exist_ok=True)

for _, row in contenu.iterrows():
    departement = row["Libellé du département"]

    inscrits = row["Inscrits"]
    votants = row["Votants"]
    blancs = row["Blancs"]
    nuls = row["Nuls"]
    exprimes = row["Exprimés"]

    abstention = inscrits - votants

    categories = ["Abstention", "Blancs", "Nuls", "Exprimés"]
    valeurs = [abstention, blancs, nuls, exprimes]
    couleurs = ["lightgray", "gold", "red", "green"]

    plt.figure(figsize=(10, 8))
    plt.pie(
        valeurs,
        labels=categories,
        autopct="%1.1f%%",
        startangle=90,
        colors=couleurs
    )
    plt.title(f"Répartition des votes - {departement}")

    nom_propre = departement.replace("/", "_")
    plt.savefig(
        f"graphiques_circulaires/{nom_propre}_circulaire.png",
        bbox_inches="tight"
    )
    plt.close()

print("Diagrammes circulaires créés avec succès")


# Question 9 : histogramme


plt.figure(figsize=(10, 6))
plt.hist(contenu["Inscrits"], bins=30)
plt.title("Distribution des inscrits par département")
plt.xlabel("Nombre d'inscrits")
plt.ylabel("Nombre de départements")
plt.grid(axis="y", alpha=0.3)

plt.savefig("histogramme_inscrits.png", bbox_inches="tight")
plt.show()

print("Histogramme créé avec succès")
