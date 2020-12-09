Ce dossier contient les fichiers Matlab de notre 1er réseau de neurones à  Convolution spécialisé pour la reconnaissance d'images.


## Constitution d'un CNN ##

# Couche de convolution
	- Objectif : Détecter une forme particulière
	- Fichiers : Conv3x3.m
		         creer_regions.m
	- Remarque : Conv3x3 est une class qui prend en entrée une ou des image(s) et un nombre de filtre de la couche deconvolution. Ce nombre correspond au nombre de neurones dans cette couche. En sortie il y autant d'images que le produit (nombre de filtres de la couche * nombre images en entrée de cette couche de convolution)

# Couche de maxpooling
	- Objectif : Eviter le surapprentissage en généralisant la localisation des formes détectées par la couche de convolution précédente
	- Fichiers : MaxPool2.m
		         creer_regions_pooling.m
	- Remarque : MaxPool2 est une class. Dans cette couche il y autant de neuronnes que dans la couche de convolution précédente. En sortie il y a autant d'images qu'en entrée.

# Couche de softmax
	- Objectif : Classer les images en leur attribuant un label
	- Fichiers : Softmax.m
	- Remarque : C'est une class qui donne en sortie un vecteur donc les valeurs représentent la probabilité d'appartenir à une classe


## Script de test  ##

# Fonction forward
	- Objectif : Traverser le réseau CNN de l'entrée vers la sortie
	- Fichiers : forward_CNN.m
	- Remarque : C'est une fonction qui donne en sortie le vecteur de sortie de la couche softmax, le loss et un boolean qui représente si la prédiction est juste ou fausse

# Fonction entrainement
	- Objectif : Entraîner le réseau CNN pour améliorer ses performances
	- Fichier  : train_CNN.m

# Script pour entraîner le réseau CNN
	- Objectif : Définir l'architecture du réseau, et l'entraîner
	- Fichier  : cnn_new_v3.m
	
# Script pour tester le réseau CNN déjà entraîné
	- Objectif : Tester les performances du réseau CNN
	- Fichier  : cnn_new_v2.m
	
# Script pour afficher sous forme d'images une représentation graphique de la vision de l'ordinateur
	- Objectif : Afficher le résultat des opérations réalisées dans les couches de Convolution et de Maxpooling
	- Fichier  : dem_une_img.m
	
# Script pour utiliser un réseau entraîné sur le Raspberry Pi
	- Objectif : Reconnaître des chiffres sur des photos de la camera du Raspberry Pi
	- Fichier  : RaspberryPi_Use_CNN.m
