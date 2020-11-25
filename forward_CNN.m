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
%% @deftypefn {} {@var{retval} =} forward_CNN (@var{input1}, @var{input2})
%%
%% @seealso{}
%% @end deftypefn

%% Author: flori <flori@LAPTOP-BLEU> & <@LAPTOP-PFAEE1RA>
%% Created: 2020-10-27

function [out,loss,acc] = forward_CNN (Conv,Pool,Softmax,image, label)
%% L'image traverse les couches
  out = forward_convolution(Conv,image);
  out = forward_pooling(Pool,out);
  out = forward_softmax(Softmax,out);
  
%% Calcul du loss 
  loss=-log(out(label)); % -log(proba(bonne classe de l'image))
  
%% Calcul de acc qui indique si le réseau a bien classé l'image
  I=find(out==max(out)); % I est le numéro prédit
  if I==label
    acc=1;
  else 
    acc=0;
  endif
  
endfunction
