classdef Softmax < handle
  properties (Access = public)
    weights
    biases
    
    last_input_size
    last_input
    last_totals
    
    test
    
    %learn_rate
    
    %%debug
    dbiais
    dweights
    debug
  end
	
  methods  
			function obj = Softmax(input_len,nodes)
        % input_len : nb éléments dans la matrice qui sort de la couche de pooling en entrée de ce Softmax
        % nodes     : nb de classes que l'on veut distinguer en sortie du réseaux CNN
        %% We divide by input_len to reduce the variance of our initial values
				obj.weights=randn(input_len,nodes)/(input_len); % Initialisation des poids
                                                           % Matrice avec autant de ligne que de pixels en entrée de la couche Softmax et autant de colonne que de classes de sortie
				obj.biases=zeros(nodes,1);  % Initialisation des biais 
                                      % Vecteur colonne avec autant de lignes que de classes de sortie
			end
			
			function retval = forward_softmax(obj,inpu)    % inpu c'est la sortie d'une couche de pooling
				[input_len,nodes]=size(obj.weights);
        flatinput=reshape(inpu,prod(size(inpu)),1); % Transforme inpu en vecteur colonne avec prod(size(inpu)) lignes
                                                       % prod(size(inpu)) = produit des dimensions de inpu par exemple 13*13*8
        %obj.test=zeros(7);
        %obj.test=inpu;
        
        totals=obj.weights' * flatinput + obj.biases;
        expto=exp(totals);
        %fprintf('%d\n',totals);
        
        % A COMMENTER APRES APPRENTISSAGE %
        % Sauvgarde de certain paramètre de l'objet pour la rétropropagation de gradient
        obj.last_input_size=size(inpu); %  Sauvgarde dans l'objet la taille de l'ancienne entrée pour rétroprog 
				obj.last_input=flatinput;       %  Sauvgarde dans l'objet l'ancienne entrée pour rétroprog
        obj.last_totals=totals;         %  Sauvgarde dans l'objet l'ancien totals pour rétroprog
				
				retval=expto/sum(expto);
			end
      
      
      function d_L_d_inputs = backprop_softmax(obj,d_L_d_out,learn_rate)
%       Dans le vecteur d_L_d_out il y a qu'une seule valeur qui est non nulle
%       Donc pour gagner du temps de traitement, on ne fait pas les calculs pour les autres qui sont nulle
        for i=1:size(d_L_d_out)
          if d_L_d_out(i)!=0
            break 
          end
        end
%       On arrive à cette ligne uniquement lorsque i cible la valeur de d_L_d_out non nulle
%       On récupère l'ancien totals du forward softmax
        t_exp=exp(obj.last_totals);
        S=sum(t_exp);
        
%       Gradients of out[i] against totals
        d_out_d_t=-t_exp(i)*t_exp/(S^2);
        d_out_d_t(i)=t_exp(i)*(S-t_exp(i))/(S^2);
        
%       Gradients of totals against weights/biases/input
        d_t_d_w = obj.last_input;
        d_t_d_b = 1;
        d_t_d_inputs = obj.weights;
        
%       Gradients of loss against totals
        d_L_d_t = d_L_d_out(i) * d_out_d_t;
        
       
        
%       Gradients of loss against weights/biases/input
        d_L_d_w = d_t_d_w * (d_L_d_t');
        d_L_d_b = d_L_d_t * d_t_d_b;
        d_L_d_inputs = d_t_d_inputs * d_L_d_t;
        
         obj.debug=d_L_d_w;
        
%       Update weights / biases
        obj.weights = obj.weights - learn_rate * d_L_d_w;
        obj.biases =  obj.biases - learn_rate * d_L_d_b;
        obj.dbiais=obj.biases;
        obj.dweights=obj.weights;
        
        d_L_d_inputs = reshape(d_L_d_inputs,obj.last_input_size);  #on remet en forme car l'entrée à été aplatie lors du forward.
      end
      
 end
end
