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
%% @deftypefn {} {@var{retval} =} MaxPool2 (@var{input1}, @var{input2})
%%
%% @seealso{}
%% @end deftypefn

%% Author: flori <flori@LAPTOP-BLEU>
%% Created: 2020-10-25

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Classe de la couche de Max Pooling %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

classdef MaxPool2 < handle
	properties (SetAccess = public, GetAccess = public)
		last_input;
		last_output;
	end
  
	methods
%%-Initialise la couche de Max Pooling-%%
	function obj = MaxPool2()
      obj.last_input = 0;  
    end
%%-------------------------------------%%

%%-Max-Pooling sur la sortie de la couche de convolution-%%
    function output = forward_pooling(obj,image)
	  % Sauvegarde l'entrée de la couche de MaxPooling
	  obj.last_input = image;
	  % Récupération du nombre et de la taille des images
	  [x,y,nb_filtres] = size(image);
	  new_x = fix(x/2);
	  new_y = fix(y/2);
	  % Initialise la sortie de la couche MaxPooling
	  output = zeros(new_x, new_y, nb_filtres);
      
	  for filtre=1:nb_filtres
	    for xx=1:new_x
	   	  for yy=1:new_y
		    % Isole une région 2x2
		    sousreg=image((xx*2)-1:(xx*2),(yy*2)-1:(yy*2),filtre);
		    % Max-Pooling sur la région 2x2
		    output(xx,yy,filtre) = max(max(sousreg));
		  end
	    end
	  end
	  % Sauvegarde la sortie pour la backprop_pooling
	  obj.last_output = output;
    end
%%-------------------------------------------------------%%
	
    %%-Matrice pour tester la class et ses fonctions-%%
    % image(:,:) = [[11,1,1,2,22,2];[1,1,1,2,2,2];[1,1,1,2,2,2];[3,3,3,4,4,4];[3,33,3,4,4,4];[3,3,3,44,4,4]]
    % image(:,:) = [[11,1,2,22];[1,1,2,2];[33,3,44,4];[3,3,4,4]]
    % image(:,:,2) = [[5,55,66,6];[5,5,6,6];[7,7,8,8];[77,7,88,8]]
    % creer_regions_pooling(image)
    % imregion=creer_regions_pooling(image)
    %%------------------------------------------------%%
    
%%-Rétropropagation du gradient du loss dans la couche de Max-Pooling-%%
    function dL_dinput = backprop_pooling(obj,dL_dout)
      [x,y,nb_filtres] = size(obj.last_input);
      dL_dinput = zeros(x,y,nb_filtres); % Initialise la sortie de la backprop_pooling
      new_x = fix(x/2);
      new_y = fix(y/2);
      
      % Parcourir les sous régions
      for filtre=1:nb_filtres
        for xx=1:new_x
          for yy=1:new_y
            amax = obj.last_output(xx,yy,filtre);
            for i=0:1
              for j=0:1
                % Compare la sortie du Max-Pooling 
				% à la valeur des pixels qui ont permis de créer la sous région
                if obj.last_input(xx*2-i,yy*2-j,filtre) == amax
                  % Si un pixel de l'image qui a permis de calculer la sortie précédente,
				  % il était un maximum dans sa sous région,
                  % Alors ce pixel a eu un impact sur l'incertitude de prédiction.
                  % Donc mettons à jour la dL_dinput
                  dL_dinput(xx*2-i,yy*2-j,filtre) = dL_dout(xx,yy,filtre);
                end
              end
            end
          end
        end
      end
    end
%%--------------------------------------------------------------------%%
	
    %%-Script permet de tester backprop_pooling-%%
    % image(:,:) = [[11,1,2,22];[1,1,2,2];[33,3,44,4];[3,3,4,4]]
    % image(:,:,2) = [[5,55,66,6];[5,5,6,6];[7,7,8,8];[77,7,88,8]]
    % p = MaxPool2
    % out_foraward = forward_pooling(p,image)
    % out_backprop = backprop_pooling(p,out_foraward)
    %%------------------------------------------%%
  end
end