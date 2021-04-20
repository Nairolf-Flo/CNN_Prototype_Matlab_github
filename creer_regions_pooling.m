%% Copyright (C) 2020 flori
%% 
%% This program is free software: you can redistribute it and/or modify it
%% under the terms of the GNU General Public License as published by
%% the Free Software Foundation, either version 3 of the License, or
%% (at your option) any later version.
%% 
%% This program is distributed in the hope that it will be useful, but
%% WITHOUT ANY WARRANTY; without even the implied warranty of
%% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%% GNU General Public License for more details.
%% 
%% You should have received a copy of the GNU General Public License
%% along with this program.  If not, see
%% <https://www.gnu.org/licenses/>.

%% -*- texinfo -*- 
%% @deftypefn {} {@var{retval} =} res_regions_pooling (@var{input1}, @var{input2})
%%
%% @seealso{}
%% @end deftypefn

%% Author: flori <flori@LAPTOP-BLEU>
%% Created: 2020-10-25

function res_regions_pooling = creer_regions_pooling (image)
  [x,y,nb_filtres]=size(image);   % Enregistre les dimensions de l'image
  new_x = fix(x/2);
  new_y = fix(y/2);
  imregionbis = cell(new_x,new_y,nb_filtres); % Création d'un tableau vide aux dimensions de l'image
  for filtre=1:nb_filtres
    for xx=1:new_x
      for yy=1:new_y
        imregionbis(xx,yy,filtre) = image((xx*2)-1:(xx*2),(yy*2)-1:(yy*2),filtre); % Création et enregistrement des régions de taille 2x2 de l'image
      end
    end
  end
  res_regions_pooling=imregionbis;
end
