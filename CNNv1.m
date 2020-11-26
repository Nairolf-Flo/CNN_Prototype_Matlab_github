#####################################################
## Script pour tester la convolution et le pooling ##
#####################################################

#####################################################
## Chargement de la base de données d'entrainement ##
#####################################################
pkg load nan
[train_data, train_labels, test_data, test_labels] = load_mnist(); # Charge la bdd images

##Image_numero = 2;
##reshapeImage = reshape(train_data(Image_numero,:,:),size(train_data(Image_numero,:,:),2),size(train_data(Image_numero,:,:),3));

###############################
## Affichage d'images la bdd ##
###############################
##IDimage = 1;
##offsetID = 0;
####Pour afficher plusieurs images sur une fenêtre 
##x = 2; # nombre images sur une ligne
##y = 2; # nombre images sur une colonne
##
###### Afficher une image
####reshapeImage = reshape(train_data(IDimage+offsetID,:,:),size(train_data(IDimage+offsetID,:,:),2),size(train_data(IDimage+offsetID,:,:),3));
####imagesc(reshapeImage)
##
#### Afficher plusieur images dans un tableau
##for j = 1 : x
##  for k = 1: y
##    reshapeImage = reshape(train_data(IDimage+offsetID,:,:),size(train_data(IDimage+offsetID,:,:),2),size(train_data(IDimage+offsetID,:,:),3));
##    subplot(x,y,offsetID+1);
##    imagesc(reshapeImage);
##    axis off
##    offsetID = offsetID + 1;
##  endfor
##endfor


################################
## 1ère Couche de Convolution ##
################################
tic
Conv1 = Conv3x3(8);
Conv1_sortie = forward_convolution(Conv1,reshapeImage);
fprintf('convolution1 : ')
toc
##Affiche les filtres 3x3 de la 1ère couche de convolution
##for j = 1 : Conv1.nombre_filtres
##  figure(1)
##    f=reshape(Conv1.filtres(:,:,j),3,3);
##    subplot(3,3,j)
##    imagesc(f);
##    title(num2str(j));
##    axis off
##endfor
##Affiche le résultat de la première couche de convolution
##for j = 1 : Conv1.nombre_filtres
##    figure(2)
##    reshapeImage = reshape(Conv1_sortie(:,:,j),size(Conv1_sortie,1),size(Conv1_sortie,2));
##    subplot(3,3,j)
##    imagesc(reshapeImage);
##    title(num2str(j));
##    axis off
##endfor

############################
## 1ère Couche de Pooling ##
############################
tic
Pool1=MaxPool2();
Pool1_sortie = forward_pooling(Pool1,Conv1_sortie);
fprintf('pooling1 : ')
toc
##Affiche le résultat de la première couche de pooling
for j = 1 : Conv1.nombre_filtres
    figure(3)
    reshapeImage = reshape(Pool1_sortie(:,:,j),size(Pool1_sortie,1),size(Pool1_sortie,2));
    subplot(3,3,j)
    imagesc(reshapeImage);
    title(num2str(j));
    axis off
end

#######################################
## 2 Couche de Convolution + Pooling ##
#######################################
tic
Conv2 = Conv3x3(3);
Conv2_sortie = forward_convolution(Conv2,Pool1_sortie);
fprintf('convolution2 : ')
toc
tic
Pool2=MaxPool2();
Pool2_sortie = forward_pooling(Pool2,Conv2_sortie);
fprintf('pooling2 : ')
toc
##Affiche le résultat de la 2 couche de convolution
##for j = 1 : Conv2.nombre_filtres*Conv1.nombre_filtres
##    figure(4)
##    reshapeImage = reshape(Conv2_sortie(:,:,j),size(Conv2_sortie,1),size(Conv2_sortie,2));
##    subplot(5,5,j)
##    imagesc(reshapeImage);
##    title(num2str(j));
##    axis off
##endfor
##Affiche le résultat de la 2 couche de pooling
for j = 1 : Conv2.nombre_filtres*Conv1.nombre_filtres
    figure(5)
    reshapeImage = reshape(Pool2_sortie(:,:,j),size(Pool2_sortie,1),size(Pool2_sortie,2));
    subplot(5,5,j)
    imagesc(reshapeImage);
    title(num2str(j));
    axis off
end

#######################################
## 3 Couche de Convolution + Pooling ##
#######################################
##tic
##Conv3 = Conv3x3(2);
##Conv3_sortie = forward_convolution(Conv3,Pool2_sortie);
##fprintf('convolution3 : ')
##toc
##tic
##Pool3=MaxPool2();
##Pool3_sortie = forward_pooling(Pool3,Conv3_sortie);
##fprintf('pooling3 : ')
##toc
####Affiche le résultat de la 2 couche de convolution
##for j = 1 : Conv3.nombre_filtres*Conv2.nombre_filtres*Conv1.nombre_filtres
##    figure(6)
##    reshapeImage = reshape(Conv3_sortie(:,:,j),size(Conv3_sortie,1),size(Conv3_sortie,2));
##    subplot(9,8,j)
##    imagesc(reshapeImage);
##    title(num2str(j));
##    axis off
##endfor
####Affiche le résultat de la 2 couche de pooling
##for j = 1 : Conv3.nombre_filtres*Conv2.nombre_filtres*Conv1.nombre_filtres
##    figure(7)
##    reshapeImage = reshape(Pool3_sortie(:,:,j),size(Pool3_sortie,1),size(Pool3_sortie,2));
##    subplot(9,8,j)
##    imagesc(reshapeImage);
##    title(num2str(j));
##    axis off
##endfor