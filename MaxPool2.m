## Copyright (C) 2020 flori
## 
## This program is free software: you can redistribute it and/or modify it
## under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
## 
## This program is distributed in the hope that it will be useful, but
## WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
## 
## You should have received a copy of the GNU General Public License
## along with this program.  If not, see
## <https://www.gnu.org/licenses/>.

## -*- texinfo -*- 
## @deftypefn {} {@var{retval} =} MaxPool2 (@var{input1}, @var{input2})
##
## @seealso{}
## @end deftypefn

## Author: flori <flori@LAPTOP-BLEU>
## Created: 2020-10-25

classdef MaxPool2 < handle
  
  properties (SetAccess = public, GetAccess = public)
    last_input;
    last_output; # A voir il faut faire un compromis entre la mémoire et le temps calcul
  end
  
  methods
    function obj = MaxPool2()
      obj.last_input = 0;  
    endfunction
  
    function output = forward_pooling(obj,image)
      obj.last_input = image;         # Sauvegarde les images en entrée de la couche de MaxPooling
      [x,y,nb_filtres] = size(image); # Récupération de la taille des images et du nombre d'images=nb_filtres
      new_x = fix(x/2);
      new_y = fix(y/2);
      output = zeros(new_x, new_y, nb_filtres); # Initialisation de la sortie de la couche MaxPooling
      
      for filtre=1:nb_filtres
        for xx=1:new_x
          for yy=1:new_y
            sousreg=image((xx*2)-1:(xx*2),(yy*2)-1:(yy*2),filtre);
            output(xx,yy,filtre) = max(max(sousreg));
          endfor
        endfor
      endfor
      obj.last_output = output;
    endfunction
    ####### Matrice pour tester la class et ses fonctions #######
    # image(:,:) = [[11,1,1,2,22,2];[1,1,1,2,2,2];[1,1,1,2,2,2];[3,3,3,4,4,4];[3,33,3,4,4,4];[3,3,3,44,4,4]]
    # image(:,:) = [[11,1,2,22];[1,1,2,2];[33,3,44,4];[3,3,4,4]]
    # image(:,:,2) = [[5,55,66,6];[5,5,6,6];[7,7,8,8];[77,7,88,8]]
    # creer_regions_pooling(image)
    # imregion=creer_regions_pooling(image)
    #######-----------------------------------------------#######
    
    function dL_dinput = backprop_pooling(obj,dL_dout)
      [x,y,nb_filtres] = size(obj.last_input);
      dL_dinput = zeros(x,y,nb_filtres); # Initialisation de la sortie de la backpropooling
      imregion = creer_regions_pooling(obj.last_input); # Création d'une matrice de sous régions 2x2 de la last_input
      [im_x,im_y,im_f] = size(imregion);
      
      # Parcourir les sous régions
      for filtre=1:im_f
        for xx=0:im_x-1
          for yy=0:im_y-1
            ###-----------------------------------------------------------------------###
            ### A décommmenter pour refaire le calcul de la sortie du forward_pooling ###
            ## Pour toutes les sous régions recalculons la sortie précédente de la couche de pooling
            #sousregion = cell2mat(imregion(xx+1,yy+1,filtre)); # Transforme la sous régions en matrice
            #amax = max(max(sousregion)); 
            ###-----------------------------------------------------------------------###
            amax = obj.last_output(xx+1,yy+1,filtre);
            for i=1:2
              for j=1:2
                # on compare cette sortie précédente à la valeur des pixels qui ont permis de créer la sous région
                if obj.last_input(xx*2+i,yy*2+j,filtre) == amax
                  # Si la valeur du pixel de l'image qui a permis de calculer la sortie précédente,
                  # était un maximum dans sa sous région,
                  # Alors ce pixel a eu un impact pour l'évaluation de la class de l'image.
                  # Donc mettons à jour la dL_dinput
                  dL_dinput(xx*2+i,yy*2+j,filtre) = dL_dout(xx+1,yy+1,filtre);
                endif
              endfor
            endfor
          endfor
        endfor
      endfor
      
    endfunction
    ### Script permet de tester backprop_pooling ###
    # image(:,:) = [[11,1,2,22];[1,1,2,2];[33,3,44,4];[3,3,4,4]]
    # image(:,:,2) = [[5,55,66,6];[5,5,6,6];[7,7,8,8];[77,7,88,8]]
    # p = MaxPool2
    # out_foraward = forward_pooling(p,image)
    # out_backprop = backprop_pooling(p,out_foraward)
    ###------------------------------------------###
    
    #Elapsed time is 0.00437617 seconds.  -> Sans stockage de la sortie précédente de forward_pooling
    #Elapsed time is 0.00143409 seconds.  -> Avec stockage de la sortie précédente de forward_pooling
    # Si pas suffisament de RAM sur l'ordinateur qui réalise l'entrainement, il faut recalculer la sortie de forward_pooling
    # Commenter les lignes 30 55 et 82
    # Décommenter les lignes 79 et 80
  end
end