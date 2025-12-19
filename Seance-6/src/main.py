#coding:utf8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import scipy.stats
import math

#Fonction pour ouvrir les fichiers
def ouvrirUnFichier(nom):
    with open(nom, "r", encoding='utf-8') as fichier: 
        contenu = pd.read_csv(fichier, low_memory=False) 
    return contenu

#Fonction pour convertir les données en données logarithmiques
def conversionLog(liste):
    log = []
    for element in liste:
        log.append(math.log(element))
    return log

#Fonction pour trier par ordre décroissant les listes (îles et populations)
def ordreDecroissant(liste):
    liste.sort(reverse = True)
    return liste

#Fonction pour obtenir le classement des listes spécifiques aux populations
def ordrePopulation(pop, etat):
    ordrepop = []
    for element in range(0, len(pop)):
        if np.isnan(pop[element]) == False:
            ordrepop.append([float(pop[element]), etat[element]])
    ordrepop = ordreDecroissant(ordrepop)
    for element in range(0, len(ordrepop)):
        ordrepop[element] = [element + 1, ordrepop[element][1]]
    return ordrepop

#Fonction pour obtenir l'ordre défini entre deux classements (listes spécifiques aux populations)
def classementPays(ordre1, ordre2):
    classement = []
    if len(ordre1) <= len(ordre2):
        for element1 in range(0, len(ordre2) - 1):
            for element2 in range(0, len(ordre1) - 1):
                if ordre2[element1][1] == ordre1[element2][1]:
                    classement.append([ordre1[element2][0], ordre2[element1][0], ordre1[element2][1]])
    else:
        for element1 in range(0, len(ordre1) - 1):
            for element2 in range(0, len(ordre2) - 1):
                if ordre2[element2][1] == ordre1[element1][1]:
                    classement.append([ordre1[element1][0], ordre2[element2][0], ordre1[element2][1]])
    return classement

#Partie sur les îles
iles = ouvrirUnFichier("./data/island-index.csv")
print(iles.columns)

surfaces = list(iles['Surface (km²)'])

# Ajout des continents
surfaces.append(float("85545323"))  # Asie / Afrique / Europe
surfaces.append(float("37856841"))  # Amérique
surfaces.append(float("7768030"))   # Antarctique
surfaces.append(float("7605049"))   # Australie

#Attention ! Il va falloir utiliser des fonctions natives de Python dans les fonctions locales que je vous propose pour faire l'exercice. Vous devez caster l'objet Pandas en list().

#Ordonner la liste obtenue avec la fonction locale ordreDecroissant() proposée.
surfaces_trie = ordreDecroissant(surfaces)

#Visualiser la loi rang-taille en créant une image de sortie.
plt.figure(figsize=(12, 6))

plt.subplot()
rangs = list(range(1, len(surfaces_trie) + 1))
plt.plot(rangs, surfaces_trie, '*')
plt.xlabel('Rang')
plt.ylabel('Surface (km²)')
plt.title('Loi rang-taille - Échelle linéaire')
plt.grid(True)
# plt.show()


rangs_log = conversionLog(rangs)
surfaces_log = conversionLog(surfaces_trie)
plt.plot(rangs, surfaces_trie, '*', markersize=3)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Rang (log)')
plt.ylabel('Surface (km²) (log)')
plt.title('Loi rang–taille - Échelle logarithmique')
plt.grid(True, which="both")
# plt.show()


"""
Est-il possible de faire un test sur les rangs ? (mettre votre réponse sous la forme d’un
commentaire dans le fichier)

Les tests de corrélation des rangs nécessitent DEUX variables pour comparer leurs classements.


sur ce jeu de données car nous n'avons qu'UNE SEULE variable (la surface).

il n'est donc pas possible de faire un test de corrélation des rangs

"""


#Partie sur les populations des États du monde
#Source. Depuis 2007, tous les ans jusque 2025, M. Forriez a relevé l'intégralité du nombre d'habitants dans chaque États du monde proposé par un numéro hors-série du monde intitulé États du monde. Vous avez l'évolution de la population et de la densité par année.
# monde = pd.DataFrame(ouvrirUnFichier("./data/Le-Monde-HS-Etats-du-monde-2007-2025.csv"))

#Attention ! Il va falloir utiliser des fonctions natives de Python dans les fonctions locales que je vous propose pour faire l'exercice. Vous devez caster l'objet Pandas en list().
monde = ouvrirUnFichier('data/Le-Monde-HS-Etats-du-monde-2007-2025.csv')


#Isoler les colonnes « État », « Pop 2007 », « Pop 2025 », « Densité 2007 » et « Densité 2025 »
etats = list(monde['État'])
pop_2007 = list(monde['Pop 2007'])
pop_2025 = list(monde['Pop 2025'])
densite_2007 = list(monde['Densité 2007'])
densite_2025 = list(monde['Densité 2025'])


pop_2007_ord = ordrePopulation(pop_2007, etats)
pop_2025_ord = ordrePopulation(pop_2025, etats)
densite_2007_ord = ordrePopulation(densite_2007, etats)
densite_2025_ord = ordrePopulation(densite_2025, etats)


classement_pop = classementPays(pop_2007_ord,pop_2025_ord)
classement_pop.sort()
print(classement_pop)

classement_dens = classementPays(densite_2007_ord,densite_2025_ord)
classement_dens.sort()
print(classement_dens)
#Isoler les deux colonnes sous la forme de liste différents en utilisant une boucle
rangs_pop_2007 = []
rangs_pop_2025 = []
dens_pop_2007 = []
dens_pop_2025 = []

for x in  classement_pop : 
    rangs_pop_2007.append(x[0])
    rangs_pop_2025.append(x[1])
    
    dens_pop_2007.append(x[0])
    dens_pop_2025.append(x[1])


from scipy.stats import spearmanr, kendalltau

print("\n--- ANALYSE POPULATION 2007 vs 2025 ---")
rho_pop, p_value_spear_pop = spearmanr(rangs_pop_2007, rangs_pop_2025)
tau_pop, p_value_kend_pop = kendalltau(rangs_pop_2007, rangs_pop_2025)

print(f"Coefficient de Spearman (ρ) : {rho_pop:.4f} (p-value : {p_value_spear_pop:.4e})")
print(f"Coefficient de Kendall (τ) : {tau_pop:.4f} (p-value : {p_value_kend_pop:.4e})")

print("\n--- ANALYSE DENSITÉ 2007 vs 2025 ---")
rho_dens, p_value_spear_dens = spearmanr(dens_pop_2007, dens_pop_2025)
tau_dens, p_value_kend_dens = kendalltau(dens_pop_2007, dens_pop_2025)

print(f"Coefficient de Spearman (ρ) : {rho_dens:.4f} (p-value : {p_value_spear_dens:.4e})")
print(f"Coefficient de Kendall (τ) : {tau_dens:.4f} (p-value : {p_value_kend_dens:.4e})")
    

