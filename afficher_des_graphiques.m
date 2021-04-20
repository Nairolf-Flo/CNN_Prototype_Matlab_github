load Filter4_evolution.mat
NBepoch = 5;

%%-Visualisation de l'�volution du filtre num�ro 4-%%
%indice = 150;
%A = zeros(3,3,NBepoch);
%for epoch=1:NBepoch
%  for y=1:3
%    for x=1:3
%      A(x,y,epoch) = r(epoch,y,x);
%    end
%  end
%  indice = indice + 1;
%  subplot(indice)
%  imagesc(A(:,:,epoch))
%end
%
%figure(2)
%r=reshape(r,NBepoch,9);
%plot(r)
%title ("�volution du filtre num�ro 4")
%ylabel("Valeur des 9 poids du filtre");
%xlabel("Epoch");
%%-------------------------------------------------%%

%%-Visualisation de l'�volution du taux de r�ussite et du loss-%%
load Accuracy_evolution.mat
load Loss_evolution

abscisse = 1 : NBepoch;
figure(3)        
plot(abscisse,q,";�volution du loss;")
leg = legend ("location", "northeast");
set (leg, "fontsize", 20);
ylabel("Loss");
xlabel("Epoch");

figure(4)     
plot(abscisse,a,";�volution du taux de succ�s;")
leg = legend ("location", "southeast");
set (leg, "fontsize", 20);
ylabel("Taux de succ�s");
xlabel("Epoch");
%%-------------------------------------------------------------%%

%%-Initialisation du r�seau-%%
load Conv1_FiltresV3.mat;
load Softmax1_weightsV3.mat;
load Softmax1_biasesV3.mat;

Conv1 = Conv3x3(size(Conv1_Filtres)(3),false);
Pool1 = MaxPool2();
Softmax1 = Softmax(size(Softmax1_weights)(1),size(Softmax1_weights)(2),false);
%%--------------------------%%

%%-Initialisation des images-%%
mnistfilenames = cell(4,1);
mnistfilenames{1} = "train-images-idx3-ubyte";
mnistfilenames{2} = "train-labels-idx1-ubyte";
mnistfilenames{3} = "t10k-images-idx3-ubyte";
mnistfilenames{4} = "t10k-labels-idx1-ubyte";
[train_data train_labels test_data test_labels]=mnistread(mnistfilenames);
%%---------------------------%%

%%-Interpr�tation graphique de la vision du CNN-%%
imgr=test_data(9999,:,:);	% r�cup�ration de l'image 9999
[d,l,L]=size(imgr); 	% pr�paration de l'image
img=reshape(imgr,l,L);
imgn=double((img/255.0))-0.5;

out = forward_convolution(Conv1,imgn);
for j=1:size(Conv1_Filtres)(3)
  figure(5)
  subplot(2,4,j)
  imagesc(out(:,:,j)) 
  titre = strcat("R�sultat du neurone de Convolution n� ", num2str(j));
  title(titre);
  axis off
end

out = forward_pooling(Pool1,out);
for j=1:size(Conv1_Filtres)(3)
  figure(6)
  subplot(2,4,j)
  imagesc(out(:,:,j)) 
  titre = strcat("R�sultat du neurone de Max-Pooling n� ", num2str(j));
  title(titre);
  axis off
end

figure(7)
imagesc(imgn)
title("Image � Identifier");
axis off
%%----------------------------------------------%%