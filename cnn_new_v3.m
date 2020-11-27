%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Script pour entraîner le réseau CNN sur un ensemble d'images %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

pkg load nan
[train_data, test_data, train_labels, test_labels] = load_mnist();% un peu lourd, charge la base de digits

%    img1=imread('im_test.jpg','jpg');  %image de test
%    img2=imread('ensil.jpg','jpg');    %image de test
%    img3=imread('ensilr2.jpg','jpg');  %image de test

%%-Images à tester-%%
offs=1;
N=1000; % nombre d'image à tester (Attention c'est vite très long)
tab_label=train_labels(offs:N+offs-1); 
tab_imgr=train_data(offs:N+offs-1,:,:); % on teste avec N images
%%-----------------%%

learn_rate=0.005; % Taux d'apprentissage

%%-Initialisation des différentes couches du réseau CNN-%%
Conv1 = Conv3x3(8);             % Couche de convolution à 8 filtres 28x28x1 -> 26x26x8
Pool1 = MaxPool2();             % Couche de Maxpooling              28x28x1 -> 26x26x8
Softmax1 = Softmax(13*13*8,10); % Couche de Softmax                 13x13x8 -> 10
%%------------------------------------------------------%%



tic

num_correct=0;
loss=0;

%%-Initialisation pour des graphiques-%%
m=1:(N+offs-1);
r=zeros(N-offs,3,3);
%%------------------------------------%%

for i=1:(N)
  label=tab_label(i);
  label=label+1;
  imgr=tab_imgr(i,:,:);
  [d,l,L]=size(imgr);
  img=reshape(imgr,l,L);
  imgn=double((img/255.0))-0.5;
  [l,acc]=train_CNN(Conv1,Pool1,Softmax1,imgn,label,learn_rate);   % Entrainement du réseau
  num_correct=num_correct+acc;
  loss=loss+l;
  
  r(i,:,:)=Conv1.filtres(:,:,4); % Enregistre les poids du filtre numero 4
 
%%-Après 100 image afficher des information sur l'apprentissage-%%
  if rem(i,100)==0  
    fprintf('Step %d : Past 100 steps: Average Loss %d | Accuracy : %d\n',i, loss/100,num_correct)
    num_correct=0;
    loss=0;
  end
%%--------------------------------------------------------------%%
end

toc

%%-Affichage de graphiques-%%
%figure(1)
%title("biais")
%plot(m,p)
%figure(2)
%title("weights")
%plot(m,q)

figure(3)
title ("filtres")
r=reshape(r,N+offs-1,9);
plot(m,r)
%%-------------------------%%