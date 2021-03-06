%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Script pour entra�ner le r�seau CNN sur un ensemble d'images %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic
mnistfilenames = cell(4,1);
mnistfilenames{1} = "train-images-idx3-ubyte";
mnistfilenames{2} = "train-labels-idx1-ubyte";
mnistfilenames{3} = "t10k-images-idx3-ubyte";
mnistfilenames{4} = "t10k-labels-idx1-ubyte";
[train_data ,train_labels, test_data, test_labels]=mnistread(mnistfilenames);
toc

%%-Images � tester-%%
offs=1;
N=60000; % nombre d'image � tester (Attention c'est vite tr�s long)
tab_label=train_labels(offs:N+offs-1); 
tab_imgr=train_data(offs:N+offs-1,:,:); % on teste avec N images
%%-----------------%%

learn_rate=0.005; % Taux d'apprentissage

%%-Initialisation des diff�rentes couches du r�seau CNN-%%
Conv1 = Conv3x3(8,true);             % Couche de convolution � 8 filtres 28x28x1 -> 26x26x8
Pool1 = MaxPool2();                  % Couche de Maxpooling              26x26x1 -> 13x13x8
Softmax1 = Softmax(13*13*8,10,true); % Couche de Softmax                 13x13x8 -> 10
%%------------------------------------------------------%%


tic

num_correct=0;
loss=0;

%%-Initialisation pour des graphiques-%%
r=zeros(N-offs,3,3);
q=zeros((N-offs+1)/100,1);
a=zeros((N-offs+1)/100,1);

%%------------------------------------%%

for i=1:(N)
  label=tab_label(i);
  label=label+1;
  imgr=tab_imgr(i,:,:);
  [d,l,L]=size(imgr);
  img=reshape(imgr,l,L);
  imgn=double((img/255.0))-0.5;
  [l,acc]=train_CNN(Conv1,Pool1,Softmax1,imgn,label,learn_rate);   % Entrainement du r�seau
  num_correct=num_correct+acc;
  loss=loss+l;
  
  
  r(i,:,:)=Conv1.filtres(:,:,4); % Enregistre les poids du filtre numero 4
 
%%-Apr�s 100 image afficher des information sur l'apprentissage-%%
  if rem(i,100)==0  
    fprintf('Step %d : Past 100 steps: Average Loss %d | Accuracy : %d\n',i, loss/100,num_correct)
    
    q(i/100,:)=double(loss/100); %Enregistre le loss moyen sur 100 images
    a(i/100,:)=num_correct;      %Enregistre le taux de succ�s sur 100 images
    
    num_correct=0;
    loss=0;
    
  end
%%--------------------------------------------------------------%%
end

toc

Conv1_Filtres =  Conv1.filtres;
Softmax1_weights = Softmax1.weights;
Softmax1_biases  = Softmax1.biases;

%%-Enregistre le r�seau entrain� dans un fichier-%%
save Conv1_Filtres.mat Conv1_Filtres;
save Softmax1_weights.mat Softmax1_weights;
save Softmax1_biases.mat Softmax1_biases;
%%-----------------------------------------------%%


figure(3)
title ("filtres")
r=reshape(r,N+offs-1,9);
plot(r)

figure(4)
title("loss and taux de succ�s")               
plot(q,'r')
yyaxis right
plot(a,'b')
%%-------------------------%%