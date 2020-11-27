%%% Supposons le réseau entraîné :

[train_data, test_data, train_labels, test_labels] = load_mnist();

N=8234;

label=train_labels(N);
label=label+1;  %label entre 1 et 10

%img1=imread('img_test.jpg','jpg');  %image de test
%imagesc(img1)
%img1=img1';

img=train_data(N,:,:);
[d,l,L]=size(img);
img=reshape(img,l,L);
imgs = img';
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