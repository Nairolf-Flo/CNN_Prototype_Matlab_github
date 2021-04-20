%%-Initialisation des images-%%
mnistfilenames = cell(4,1);
mnistfilenames{1} = "train-images-idx3-ubyte";
mnistfilenames{2} = "train-labels-idx1-ubyte";
mnistfilenames{3} = "t10k-images-idx3-ubyte";
mnistfilenames{4} = "t10k-labels-idx1-ubyte";
[train_data train_labels test_data test_labels]=mnistread(mnistfilenames);
%%---------------------------%%

%%-Images à tester-%%
offs=1;
N=100; % nombre d'image à tester (Attention c'est vite très long)
tab_label=test_data(offs:N+offs-1); 
tab_imgr=test_labels(offs:N+offs-1,:,:); % on teste avec N images
%%-----------------%%

%%-Initialisation du réseau-%%
load Conv1_FiltresV3.mat;
load Softmax1_weightsV3.mat;
load Softmax1_biasesV3.mat;

Conv1 = Conv3x3(size(Conv1_Filtres)(3),false);
Pool1 = MaxPool2();
Softmax1 = Softmax(size(Softmax1_weights)(1),size(Softmax1_weights)(2),false);
%%--------------------------%%
  
%%-Analyse l'image avec le CNN-%%
cnt_acc = 0;
loss = 0;
for i=offs:N
  label=test_labels(i);	% récupération du label i
  label=label+1;		% map le label entre 1 et 10
  imgr=test_data(i,:,:);	% récupération de l'image i
  [d,l,L]=size(imgr); 	% préparation de l'image
  img=reshape(imgr,l,L);
  imgn=double((img/255.0))-0.5;
  [out,l,acc]=forward_CNN(Conv1,Pool1,Softmax1,imgn,label);
  
  cnt_acc = cnt_acc + acc;
  loss = loss + l;
  fprintf("Prédiction %d | Réalité %d \n",find(out==max(out))-1, label-1)
end

fprintf("Taux de succès %d | loss moyen %d\n", (cnt_acc*100) / (N-offs+1), loss / (N-offs+1))