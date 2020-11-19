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
## @deftypefn {} {@var{retval} =} creer_regions (@var{input1}, @var{input2})
##
## @seealso{}
## @end deftypefn

## Author: flori <flori@LAPTOP-BLEU>
## Created: 2020-10-25

######################################################
## Fonction utilisée dans une couche de convolution ##
## pour réaliser l'opération de convolution         ##
######################################################
function regions = creer_regions (image)
  [x,y,p]=size(image);       # Enregistre les dimensions image
  imregionbis = cell(x,y,p);# Création d'un tableau vide aux dimensions de l'image
  for pp=1:p
    for xx=1:(x-2)
      for yy=1:(y-2)
        imregionbis(xx,yy,pp) = image(xx:(xx+2), yy:(yy+2), pp); # Création et enregistrement des régions de taille 3x3 de l'image
      endfor
    endfor
  endfor
  regions=imregionbis;
endfunction
