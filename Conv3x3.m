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
%% along with obj program.  If not, see
%% <https://www.gnu.org/licenses/>.

%% -*- texinfo -*- 
%% @deftypefn {} {@var{retval} =} Conv3x3 (@var{input1}, @var{input2})
%%
%% @seealso{}
%% @end deftypefn

%% Author: flori <flori@LAPTOP-BLEU>
%% Created: 2020-10-21

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Classe de la couche de Convolution %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
classdef Conv3x3 < handle
  properties (SetAccess = public, GetAccess = public)
    filtres
    nombre_filtres
    last_input
  end
  
  methods
    
    %%-Initialisation des nb_filtres filtres-%%
    function obj=Conv3x3(nb_filtres,bool)
      obj.nombre_filtres=nb_filtres;
      if bool==true
        obj.filtres=randn([3,3,nb_filtres])./9;
      else
        load Conv1_FiltresV3.mat;
        obj.filtres=Conv1_Filtres;
      end
    end
    %%---------------------------------------%%

	%%-Convolution entre l'image et les filtres-%%
    function resultat_convolution=forward_convolution(obj,image)
      [h,l,p]=size(image);
	  % Initialise la matrice résultat
      resultat_convolution=zeros(h-2,l-2,obj.nombre_filtres,p);
      
      for pp=1:p % p = 1 si Conv3x3 est la 1ère couche du CNN et que l'image est sur un canal
        for i=1:(h-2)
          for j=1:(l-2)
            matx=image(i:(i+2), j:(j+2), pp); % Isole une région de l'image de dimension 3x3
            resultat_convolution(i,j,:,pp)=sum(sum(matx.*obj.filtres)); % Convolution
          end
        end
      end
      % Sauvgarde dans l'objet l'ancienne entrée pour rétropropagation
      obj.last_input = image; 
      
    end
	%%------------------------------------------%%
    
	%%-Rétropropagation du gradient du loss dans la couche de convolution-%%
    function none=backprop_conv(obj,d_L_d_out,learn_rate)
          none=0;
          d_L_d_filters = zeros(size(obj.filtres)); % Initialise une Matrice 3x3
          [h,l,p]=size(obj.last_input);
          
          for pp=1:p % p = nombre de filtres
            for i=1:(h-2)
              for j=1:(l-2)
                  d_L_d_filters(:,:,:)=d_L_d_filters(:,:,:)+d_L_d_out(i,j,:).*obj.last_input(i:i+2,j:j+2);
              end
            end
          end
          
        obj.filtres =obj.filtres - learn_rate * d_L_d_filters; % MAJ des filtres
    end
	%%--------------------------------------------------------------------%%
 end
end 
  
%%-Script pour tester le forward_convolution-%%
%fi = Conv3x3(2,true)
%A = [[1,1,1,2,2,2];[1,1,1,2,2,2];[1,1,1,2,2,2];[3,3,3,4,4,4];[3,3,3,4,4,4];[3,3,3,4,4,4]]
%forward_convolution(fi,A)
%%-------------------------------------------%%