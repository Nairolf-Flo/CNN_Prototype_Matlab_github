%%% Supposons le réseau entraîné :

[train_data, test_data, train_labels, test_labels] = load_mnist();

N=1334;

label=train_labels(N);
label=label+1;  %label entre 1 et 10

img=train_data(N,:,:);
[d,l,L]=size(img);
img=reshape(img,l,L);
img=double((img/255.0))-0.5;
imagesc(img)

tic
[out,l,acc]=forward_CNN(Conv1,Pool1,Softmax1,img,label)
toc
label=label-1 % affiche le vrai label (entre 0 et 9)