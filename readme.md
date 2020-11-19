Ce dossier contient les fichiers Matlab de notre 1er rÃ©seau de neurones Ã  Convolution spÃ©cialisÃ© pour la reconnaissance d'images.

###########################
## Constitution d'un CNN ##
###########################
# Couche de convolution
	- Objectif : DÃ©tecter une forme particuliÃ¨re
	- Fichiers : Conv3x3.m
		     creer_regions.m
	- Remarque : Conv3x3 est une class qui prend en entrÃ©e une ou des image(s) et un nombre de filtre de la couche deconvolution. Ce nombre correspond au nombre de neurones dans cette couche. En sortie il y autant d'images que le produit (nombre de filtres de la couche * nombre images en entrÃ©e de cette couche de convolution)

# Couche de maxpooling
	- Objectif : Eviter le surapprentissage en gÃ©nÃ©ralisant la localisation des formes dÃ©tectÃ©es par la couche de convolution prÃ©cÃ©dente
	- Fichiers : MaxPool2.m
		     creer_regions_pooling.m
	- Remarque : MaxPool2 est une class. Dans cette couche il y autant de neuronnes que dans la couche de convolution prÃ©cÃ©dente. En sortie il y a autant d'images qu'en entrÃ©e.

# Couche de softmax
	- Objectif : Classer les images en leur attribuant un label
	- Fichiers : Softmax.m
	- Remarque : C'est une class qui donne en sortie un vecteur donc les valeurs reprÃ©sentent la probabilitÃ© d'appartenir Ã  une classe

#####################
## Script de test  ##
#####################
# Fonction forward
	- Objectif : Traverser le rÃ©seau CNN de l'entrÃ©e vers la sortie
	- Fichiers : forward_CNN.m
	- Remarque : C'est une fonction qui dÃ©finit l'architecture du rÃ©seau CNNet donne en sortie le vecteur de sortie de la couche softmax, le loss et un boolean qui reprÃ©sente si la prÃ©diction est juste ou fausse


##############
## Version  ##
##############
21:25 30/10/2020 : Correction de cnn_new_v1 et forward_CNN. Un affichage permet maintenant de visualiser la proposition de numÃ©ro pour chaque image.
hh:mm 31/10/2020 : Ajout de la rÃ©tropropagation dans Softmax, crÃ©ation de la fonction train_CNN et du fichier cnn_new_v3 pour tester le backprop
22:15 04/11/2020 : Rétropropagation de Softmax fonctionnelle !! + corrections ds divers fichiers
20:12 10/11/2020 : rÃ©tropropagation de Conv : non fonctionelle ! Ã  corriger