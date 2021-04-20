%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Script pour entraîner le réseau CNN sur un ensemble d'images %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic
mnistfilenames = cell(4,1);
mnistfilenames{1} = "train-images-idx3-ubyte";
mnistfilenames{2} = "train-labels-idx1-ubyte";
mnistfilenames{3} = "t10k-images-idx3-ubyte";
mnistfilenames{4} = "t10k-labels-idx1-ubyte";
[train_data train_labels test_data test_labels]=mnistread(mnistfilenames);
toc

%    img1=imread('im_test.jpg','jpg');  %image de test
%    img2=imread('ensil.jpg','jpg');    %image de test
%    img3=imread('ensilr2.jpg','jpg');  %image de test

%%-Images à tester-%%
offs=1;
N=200; % nombre d'image à tester (Attention c'est vite très long)
tab_label=train_labels(offs:N+offs-1); 
tab_imgr=train_data(offs:N+offs-1,:,:); % on teste avec N images
%%-----------------%%

learn_rate=0.005; % Taux d'apprentissage

%%-Initialisation des différentes couches du réseau CNN-%%
Conv1 = Conv3x3(8,true);             % Couche de convolution à 8 filtres 28x28x1 -> 26x26x8
Pool1 = MaxPool2();                  % Couche de Maxpooling              26x26x1 -> 13x13x8
Softmax1 = Softmax(13*13*8,10,true); % Couche de Softmax                 13x13x8 -> 10
%%------------------------------------------------------%%


%%-Initialisation pour des graphiques-%%
NB_epoch = 2;
r=zeros(NB_epoch,3,3);
q=zeros(NB_epoch,1);
a=zeros(NB_epoch,1);
%%------------------------------------%%

for epoch=1:NB_epoch
tic

  num_correct = 0; % Initialisation du compteur de prédiction correcte
  loss = 0;		 % Initialisation du loss

  %%-Initialisation pour des graphiques-%%
  %r=zeros((N-offs+1)/100,3,3);
  %q=zeros((N-offs+1)/100,1);
  %a=zeros((N-offs+1)/100,1);
  %%------------------------------------%%

  for i=1:(N)
    label=tab_label(i);	% récupération du label i
    label=label+1;		% map le label entre 1 et 10
    imgr=tab_imgr(i,:,:);	% récupération de l'image i
    [d,l,L]=size(imgr); 	% préparation de l'image
    img=reshape(imgr,l,L);
    imgn=double((img/255.0))-0.5;
    [l,acc]=train_CNN(Conv1,Pool1,Softmax1,imgn,label,learn_rate);  % Entrainement du réseau
    num_correct=num_correct+acc;  % MAJ du compteur de bonne prédiction
    loss=loss+l;  % MAJ du loss
    
    %%- Enregistrements pour les graphiques-%%
    %q(i,:)=l; 						% Enregistre le loss moyen des 100 premières images
    %r(i,:,:)=Conv1.filtres(:,:,3);	% Enregistre les poids du filtre numero 4
    %%-----------------------------------%%
   
  %%-Après 100 image afficher des informations sur l'apprentissage-%%
  %  if rem(i,100)==0  
  %    %fprintf('Step %d : Past 100 steps: Average Loss %d | Accuracy : %d\n',i, loss/100,num_correct)
  %  fprintf('Images %d : \nPour les 100 dernières images: Loss moyen %d | Prédictions justes : %d\n',i, loss/100,num_correct)
  %    %%- Enregistrements pour les graphiques-%%
  %  q(i/100,:)=double(loss/100); %Enregistre le loss moyen sur 100 images
  %    a(i/100,:)=num_correct;      %Enregistre le taux de succès sur 100 images
  %  r(i/100,:,:)=Conv1.filtres(:,:,3); % Enregistre les poids du filtre numero 4
  %    %%-----------------------------------%%
  %    num_correct=0;	% Rénitialisation du compteur de prédiction correcte pour les 100 prochaines images
  %    loss=0; 		% Rénitialisation du loss pour les 100 prochaines images
  %  end
  %%--------------------------------------------------------------%%
  end

  toc
  fprintf('Fin epoch %d \n',epoch)
  %%-Après 1 epoch afficher des informations sur l'apprentissage-%%
  fprintf('Loss moyen %d | Prédictions justes : %d \n', loss/(N-offs+1),(num_correct * 100) / (N-offs+1))
  %%--------------------------------------------------------------%%
  
  %%- Enregistrements pour les graphiques-%%
  q(epoch,:)=double(loss/(N-offs+1)); %Enregistre le loss moyen sur N-offs images
  a(epoch,:)=(num_correct * 100) / (N-offs+1);      %Enregistre le taux de succès sur N-offs images
  r(epoch,:,:)=Conv1.filtres(:,:,3); % Enregistre les poids du filtre numero 4
  %%-----------------------------------%%

end

%%-Enregistre le réseau entrainé dans des fichiers-%%
Conv1_Filtres =  Conv1.filtres;
Softmax1_weights = Softmax1.weights;
Softmax1_biases  = Softmax1.biases;

save Conv1_FiltresV3.mat Conv1_Filtres;
save Softmax1_weightsV3.mat Softmax1_weights;
save Softmax1_biasesV3.mat Softmax1_biases;
%%-------------------------------------------------%%


%%-Afficher les graphiques-%%
%figure(3)
%r=reshape(r,(N+offs-1)/100,9);
%plot(r)
%title ("Évolution du filtre numéro 4")
%ylabel("Valeur des 9 poids du filtre");
%xlabel("Epoch");

%abscisse = offs : (N+offs)/100;
%figure(4)
%subplot(211);           
%plot(abscisse,q,";Évolution du loss;")
%leg = legend ("location", "northeast");
%set (leg, "fontsize", 20);
%ylabel("Loss");
%xlabel("Epoch");

%subplot(212);     
%plot(abscisse,a,";Évolution du taux de succès;")
%leg = legend ("location", "southeast");
%set (leg, "fontsize", 20);
%ylabel("Taux de succès");
%xlabel("Epoch");
%%-------------------------%%

save Loss_evolution.mat q;
save Accuracy_evolution.mat a;
save Filter4_evolution.mat r;