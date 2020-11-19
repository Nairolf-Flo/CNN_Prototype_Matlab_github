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
## along with obj program.  If not, see
## <https://www.gnu.org/licenses/>.

## -*- texinfo -*- 
## @deftypefn {} {@var{retval} =} Conv3x3 (@var{input1}, @var{input2})
##
## @seealso{}
## @end deftypefn

## Author: flori <flori@LAPTOP-BLEU>
## Created: 2020-10-21

classdef Conv3x3 < handle
  properties (SetAccess = public, GetAccess = public)
    filtres
    nombre_filtres
    last_input
  end
  
  methods
    
    # Création d'une matrice avec les nombre_filtres filtres
    # utilisation : fi = Conv3x3(8);fi.filters(:,:,filtre numero i);
    function obj=Conv3x3(nb_filtres)
      obj.nombre_filtres=nb_filtres;
      obj.filtres=randn([3,3,nb_filtres])./9;
      #obj.filtres=randi([1,5],3,3,nb_filtres); # Pour tester si fonctionne bien
    endfunction
    
    function resultat_convolution=forward_convolution(obj,image)
      [h,l,p]=size(image);
##      resultat_convolution=zeros(h-2,l-2,obj.nombre_filtres); # Création d'une matrice vide contenant les images après la convolution
      resultat_convolution=zeros(h-2,l-2,obj.nombre_filtres,p); # Création d'une matrice vide contenant les images après la convolution
      
      for pp=1:p
        for i=1:(h-2)
          for j=1:(l-2)
            matx=image(i:(i+2), j:(j+2), pp);
            resultat_convolution(i,j,:,pp)=sum(sum(matx.*obj.filtres)); # Convolution des images par les filtres de la couche Conv3x3
          endfor
        endfor
      endfor
      
      obj.last_input = image;   %%  Sauvgarde dans l'objet l'ancienne entrée pour rétroprog
      
    endfunction
    
    
function none=backprop_conv(obj,d_L_d_out,learn_rate)
      none=0;
      d_L_d_filters = zeros(size(obj.filtres));
      
      [h,l,p]=size(obj.last_input);
      
      for pp=1:p
        for i=1:(h-2)
          for j=1:(l-2)
              d_L_d_filters(:,:,:)=d_L_d_filters(:,:,:)+d_L_d_out(i,j,:).*obj.last_input(i:i+2,j:j+2);
          endfor
        endfor
      endfor
      
     % Update filters
    obj.filtres =obj.filtres - learn_rate * d_L_d_filters;
    end
      
#fi = Conv3x3(2)
#A = [[1,1,1,2,2,2];[1,1,1,2,2,2];[1,1,1,2,2,2];[3,3,3,4,4,4];[3,3,3,4,4,4];[3,3,3,4,4,4]]
  end
end
