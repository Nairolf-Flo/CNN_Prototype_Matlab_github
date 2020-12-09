%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Script pour tester le réseau CNN sur un ensemble d'images %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tic
mnistfilenames = cell(4,1);
mnistfilenames{1} = "train-images-idx3-ubyte";
mnistfilenames{2} = "train-labels-idx1-ubyte";
mnistfilenames{3} = "t10k-images-idx3-ubyte";
mnistfilenames{4} = "t10k-labels-idx1-ubyte";
[train_data, train_labels, test_data, test_labels]=mnistread(mnistfilenames);
toc

%pkg load nan
%[train_data, test_data, train_labels, test_labels] = load_mnist();% un peu lourd, charge la base de digits

%%    img1=imread('im_test.jpg','jpg');  %image de test
%%    img2=imread('ensil.jpg','jpg');    %image de test
%%    img3=imread('ensilr2.jpg','jpg');  %image de test

%%-Images à tester-%%
offs=1;
N=1000; % nombre d'image à tester (Attention c'est vite très long)
tab_label=train_labels(offs:N+offs-1); 
tab_imgr=train_data(offs:N+offs-1,:,:); % on teste avec N images
%%-----------------%%

load Conv1_Filtres_60k.mat; %Entraînement sur 60000 images
load Softmax1_weights_60k.mat;
load Softmax1_biases_60k.mat;

% load Conv1_Filtres.mat;   %Entraînement selon cnn_new_v3 
% load Softmax1_weights.mat;
% load Softmax1_biases.mat;

%%-Initialisation du réseau-%%
taille_filtres=size(Conv1_Filtres);
taille_weigts=size(Softmax1_weights);
Conv1 = Conv3x3(taille_filtres(3),false);
Pool1 = MaxPool2();
Softmax1 = Softmax(taille_weigts(1),taille_weigts(2),false);

%%--------------------------%%


fprintf("Start\n")


num_correct=0;
loss=0;
tic
%stepdiv10 = 1;
for i=1:(N)
  label=tab_label(i);
  label=label+1;
  imgr=tab_imgr(i,:,:);
  [d,l,L]=size(imgr);
  img=reshape(imgr,l,L);
  imgn=double((img/255.0))-0.5;
  [out,l,acc]=forward_CNN(Conv1,Pool1,Softmax1,imgn,label);
  num_correct=num_correct+acc;
  loss=loss+l;
 
  fprintf('Proposition : %d <:::> Label : %d <:::> %d\n', find(out==max(out)),label,acc)
  
%  if rem(i,10)==0  %reste de la division par 10
%    fprintf('Pour %d images : Average Loss %d | Predictions justes : %d\n',i,loss/(10*stepdiv10),num_correct)
%    stepdiv10 = stepdiv10 +1;
%  endif
end
toc
fprintf("###-------------------------------------------------------###\n")
fprintf("Bilan pour les %d images\n",N)
pourcent_reussite=(num_correct/(N))*100
avgLoss=loss/(N)
