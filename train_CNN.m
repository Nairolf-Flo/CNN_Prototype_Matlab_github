%% Copyright (C) 2020 samsa
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
%% @deftypefn {} {@var{retval} =} train_CNN (@var{input1}, @var{input2})
%%
%% @seealso{}
%% @end deftypefn

%% Author: samsa <samsa@LAPTOP-PFAEE1RA>
%% Created: 2020-10-31

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Fonction pour entra�ner le r�seau � classer des images %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [loss,acc] = train_CNN (Conv,Pool,Softmax,image, label,learn_rate)
	% Conv : nom de la couche de convolution
	% Pool : nom de la couche de max-pooling
	% Softmax : nom de la couche de softmax
	% image : nom de l'image � classer
	% label : chiffre r�ellement dessin� sur l'image
	% learn_rate : valeur du taux d'apprentissage
	
	%% L'image est analys�e par le CNN
	[out,loss,acc]=forward_CNN(Conv,Pool,Softmax,image,label);

	%% Initialise le gradient
	gradient = zeros(10,1); % Vecteur pour clesser l'images parmi 10 cat�gories
	gradient(label) = -1 / out(label); 

	%%-R�tropropagation du gradient � travers les 3 couches du CNN-%%
	gradient=backprop_softmax(Softmax,gradient,learn_rate);
	gradient=backprop_pooling(Pool,gradient);
	gradient=backprop_conv(Conv,gradient,learn_rate);
	%%-------------------------------------------------------------%%
end
