#coding:utf8

import pandas as pd
import math
import scipy
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import shapiro

#C'est la partie la plus importante dans l'analyse de données. D'une part, elle n'est pas simple à comprendre tant mathématiquement que pratiquement. D'autre, elle constitue une application des probabilités. L'idée consiste à comparer une distribution de probabilité (théorique) avec des observations concrètes. De fait, il faut bien connaître les distributions vues dans la séance précédente afin de bien pratiquer cette comparaison. Les probabilités permettent de définir une probabilité critique à partir de laquelle les résultats ne sont pas conformes à la théorie probabiliste.
#Il n'est pas facile de proposer des analyses de données uniquement dans un cadre univarié. Vous utiliserez la statistique inférentielle principalement dans le cadre d'analyses multivariées. La statistique univariée est une statistique descriptive. Bien que les tests y soient possibles, comprendre leur intérêt et leur puissance d'analyse dans un tel cadre peut être déroutant.
#Peu importe dans quelle théorie vous êtes, l'idée de la statistique inférentielle est de vérifier si ce que vous avez trouvé par une méthode de calcul est intelligent ou stupide. Est-ce que l'on peut valider le résultat obtenu ou est-ce que l'incertitude qu'il présente ne permet pas de conclure ? Peu importe également l'outil, à chaque mesure statistique, on vous proposera un test pour vous aider à prendre une décision sur vos résultats. Il faut juste être capable de le lire.

#Par convention, on place les fonctions locales au début du code après les bibliothèques.
def ouvrirUnFichier(nom):
    with open(nom, "r") as fichier:
        contenu = pd.read_csv(fichier)
    return contenu


#Théorie de l'échantillonnage (intervalles de fluctuation)
#L'échantillonnage se base sur la répétitivité.


donnees = pd.DataFrame(ouvrirUnFichier("./data/Echantillonnage-100-Echantillons.csv"))
print(donnees.head(100))

#Calcul des moyennes des colonnes et arrondis en entier
moyennes = donnees.mean().round(0).astype(int)
print("Moyennes : ", moyennes)

#Calcul des fréquences
somme_moyennes = moyennes.sum()
print("Somme des moyennes : ",somme_moyennes)
freq_echantillon = (moyennes / somme_moyennes).round(2)
print("Fréquence : ",freq_echantillon)


# Fréquences de la population mère

pop_mere = pd.Series({
    "Pour": 852,
    "Contre": 911,
    "Sans opinion": 422
})

somme_pop = pop_mere.sum()
freq_population = (pop_mere / somme_pop).round(2)

print("\nFréquences de la population mère :")
print(freq_population)


# Comparaison

comparaison = pd.DataFrame({
    "Moyennes_echantillon": moyennes,
    "Frequences_echantillon": freq_echantillon,
    "Population_mere": pop_mere,
    "Frequences_population": freq_population
})

print("\nTableau comparatif :")
print(comparaison)

# comparaison.plot()
# plt.show()


# Paramètres pour l’intervalle de fluctuation

z = 1.96
n = somme_moyennes  # taille d'un échantillon


# Calcul des intervalles à 95 %

marges = z * np.sqrt(freq_echantillon * (1 - freq_echantillon) / n)

interval_basse = freq_echantillon - marges
interval_haute = freq_echantillon + marges

# On arrondit à 2 décimales comme demandé
interval_basse = interval_basse.round(2)
interval_haute = interval_haute.round(2)


# 6. Mise en tableau

resultat = pd.DataFrame({
    "Freq_echantillon": freq_echantillon.round(2),
    "Borne_inf_95%": interval_basse,
    "Borne_sup_95%": interval_haute
})

print("Résultat sur le calcul d'un intervalle de fluctuation")
print(resultat)


#Théorie de l'estimation (intervalles de confiance)
#L'estimation se base sur l'effectif.


# On récupère la première ligne (échantillon n°1) dans une liste
echantillon_1 = list(donnees.iloc[0])
print("Échantillon 1 :", echantillon_1)


# Calcul des effectifs et des fréquences


# Somme totale de l'échantillon
n1 = sum(echantillon_1)

# Fréquences
freq1 = [x / n1 for x in echantillon_1]

print("Taille de l'échantillon :", n1)
print("Fréquences :", freq1)


# Intervalle de confiance (IC) à 95 % pour chaque proportion

z = 1.96

bornes_inf = []
bornes_sup = []

for p in freq1:
    marge = z * np.sqrt(p * (1 - p) / n1)
    bornes_inf.append(round(p - marge, 2))
    bornes_sup.append(round(p + marge, 2))

print("Bornes inférieures à 95% :", bornes_inf)
print("Bornes supérieures à 95% :", bornes_sup)


# Tableau résultat


resultat_IC = pd.DataFrame({
    "Effectif": echantillon_1,
    "Fréquence": [round(f, 2) for f in freq1],
    "IC_95_inf": bornes_inf,
    "IC_95_sup": bornes_sup
}, index=["Pour", "Contre", "SansOpinion"])

print("Résultat sur le calcul d'un intervalle de confiance")
print(resultat_IC)

##########################################################################
#Théorie de la décision (tests d'hypothèse)
#La décision se base sur la notion de risques alpha et bêta.
#Comme à la séance précédente, l'ensemble des tests se trouve au lien : https://docs.scipy.org/doc/scipy/reference/stats.html
##########################################################################

# -------------------------------------------------------------------
# Le test de Shapiro-Wilk renvoie :
# - statistique W
# - p-value
#
# Règle de décision classique :
# Si p-value < 0.05 → on rejette l’hypothèse de normalité
# Si p-value ≥ 0.05 → on ne rejette pas la normalité (donc distribution compatible avec une loi normale)
# -------------------------------------------------------------------

print("Théorie de la décision")

# Lecture des deux fichiers

df1 = pd.read_csv("./data/Loi-normale-Test-1.csv")
df2 = pd.read_csv("./data/Loi-normale-Test-2.csv")

# Conversion en tableaux simples
serie1 = df1['Test']
serie2 = df2['Test']

print(serie1)

# Test de Shapiro-Wilk

stat1, p1 = shapiro(np.array(serie1))
stat2, p2 = shapiro(np.array(serie2))

print("Test 1 : W =", stat1, ", p-value =", p1)
print("Test 2 : W =", stat2, ", p-value =", p2)

# Interprétation automatique
alpha = 0.05

if p1 >= alpha:
    print("→ Le Test 1 est compatible avec une loi normale.")
else:
    print("→ Le Test 1 n'est pas compatible avec une loi normale.")

if p2 >= alpha:
    print("→ Le Test 2 est compatible avec une loi normale.")
else:
    print("→ Le Test 2 n'est pas compatible avec une loi normale.")
    

# CONCLUSION : Aucun des deux fichiers ne correspond à une loi normale.


