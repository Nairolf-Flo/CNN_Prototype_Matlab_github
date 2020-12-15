%%% Supposons le réseau entraîné :

tic
mnistfilenames = cell(4,1);
mnistfilenames{1} = "train-images-idx3-ubyte";
mnistfilenames{2} = "train-labels-idx1-ubyte";
mnistfilenames{3} = "t10k-images-idx3-ubyte";
mnistfilenames{4} = "t10k-labels-idx1-ubyte";
[train_data ,train_labels, test_data, test_labels]=mnistread(mnistfilenames);
toc

%pkg load nan
%[train_data, test_data, train_labels, test_labels] = load_mnist();% un peu lourd, charge la base de digits

%%    img1=imread('im_test.jpg','jpg');  %image de test
%%    img2=imread('ensil.jpg','jpg');    %image de test
%%    img3=imread('ensilr2.jpg','jpg');  %image de test

% load Conv1_Filtres_60k.mat; %Entraînement sur 60000 images
% load Softmax1_weights_60k.mat;
% load Softmax1_biases_60k.mat;

load Conv1_Filtres.mat;   %Entraînement selon cnn_new_v3 
load Softmax1_weights.mat;
load Softmax1_biases.mat;

%%-Initialisation du réseau-%%
taille_filtres=size(Conv1_Filtres);
taille_weigts=size(Softmax1_weights);
Conv1 = Conv3x3(taille_filtres(3),false);
Pool1 = MaxPool2();
Softmax1 = Softmax(taille_weigts(1),taille_weigts(2),false);

%%--------------------------%%

fprintf("Start\n")


N=1234;

label=train_labels(N);
label=label+1;  %label entre 1 et 10

%img1=imread('img_test.jpg','jpg');  %image de test
%imagesc(img1)
%img1=img1';

img=train_data(N,:,:);
[d,l,L]=size(img);
img=reshape(img,l,L);
imgs = img;
imagesc(imgs)
img=double((img/255.0))-0.5;


tic
[out,l,acc]=forward_CNN(Conv1,Pool1,Softmax1,img,label)
toc
label=label-1 % affiche le vrai label (entre 0 et 9)

out = forward_convolution(Conv1,img);
for j = 1 : 4
    figure(2)
    title("Convolution")
    subplot(2,2,j)
    imagesc(out(:,:,j)) 
    top=max(max(max(out)));
    bottom=min(min(min(out)));
    caxis manual
    caxis([bottom top]);
    axis off
end

out = forward_pooling(Pool1,out);
for j = 1 : 4
    figure(3)
    title("Pooling")
    subplot(2,2,j)
    imagesc(out(:,:,j)) 
    top=max(max(max(out)));
    bottom=min(min(min(out)));
    caxis manual
    caxis([bottom top]);
    axis off
end

for j = 1 : 4
    figure(4)
    title("Filtres")
    subplot(2,2,j)
    imagesc(Conv1.filtres(:,:,j)) 
    top=max(max(max(Conv1.filtres)));
    bottom=min(min(min(Conv1.filtres)));
    caxis manual
    caxis([bottom top]);
    axis off
end