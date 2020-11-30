%%% Supposons le réseau entraîné :

tic
mnistfilenames = cell(4,1);
mnistfilenames{1} = "train-images-idx3-ubyte";
mnistfilenames{2} = "train-labels-idx1-ubyte";
mnistfilenames{3} = "t10k-images-idx3-ubyte";
mnistfilenames{4} = "t10k-labels-idx1-ubyte";
[train_data train_labels test_data test_labels]=mnistread(mnistfilenames);
toc

%pkg load nan
%[train_data, test_data, train_labels, test_labels] = load_mnist();% un peu lourd, charge la base de digits

%%    img1=imread('im_test.jpg','jpg');  %image de test
%%    img2=imread('ensil.jpg','jpg');    %image de test
%%    img3=imread('ensilr2.jpg','jpg');  %image de test


load Conv1_Filtres.mat;
load Softmax1_weights.mat;
load Softmax1_biases.mat;

%%-Initialisation du réseau-%%
Conv1 = Conv3x3(size(Conv1_Filtres)(3),false);
Pool1 = MaxPool2();
Softmax1 = Softmax(size(Softmax1_weights)(1),size(Softmax1_weights)(2),false);
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
for j = 1 : 8
    figure(2)
    subplot(2,4,j)
    imagesc(out(:,:,j)) 
    title(num2str(j));
    axis off
end

out = forward_pooling(Pool1,out);
for j = 1 : 8
    figure(3)
    subplot(2,4,j)
    imagesc(out(:,:,j)) 
    title(num2str(j));
    axis off
end