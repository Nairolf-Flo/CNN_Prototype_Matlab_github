pkg load image

fprintf("Start\n")

while (true)
  tic
  %%-Préparation de l'image à analyser-%%
  %imgr=imread('img_test-1.jpg','jpg'); % Charge l'image
  run("camera.cpp");
  imgr=imread('photo.jpg','jpg'); % Charge l'image
  imgr=imresize(imgr,[28,28]);
  [d,l,L]=size(imgr);             % Charge les dimensions de l'image
  %img=imgr;
  [img,map] = rgb2ind(imgr);     % Modifie l'image pour la mettre sur 1 channel comme l'entraînement https://octave.org/doc/v4.0.3/Representing-Images.html
  img=reshape(img,d,l);
  imgn=double((img/255.0))-0.5;
  %%-----------------------------------%%

  %%-Initialisation du réseau-%%
  load Conv1_Filtres.mat;
  load Softmax1_weights.mat;
  load Softmax1_biases.mat;
  Conv1 = Conv3x3(size(Conv1_Filtres)(3),false);
  Pool1 = MaxPool2();
  Softmax1 = Softmax(size(Softmax1_weights)(1),size(Softmax1_weights)(2),false);
  %%--------------------------%%

  %%-Analyse l'image avec le CNN-%%
  [out,l,acc]=forward_CNN(Conv1,Pool1,Softmax1,imgn,1);
  
%  out
  l
  find(out==max(out))-1
  %%-----------------------------%%
  toc
end

%out = forward_convolution(Conv1,imgn);
%for j = 1 : 8
%    figure(2)
%    subplot(2,4,j)
%    imagesc(out(:,:,j)) 
%    title(num2str(j));
%    axis off
%end
%
%out = forward_pooling(Pool1,out);
%for j = 1 : 8
%    figure(3)
%    subplot(2,4,j)
%    imagesc(out(:,:,j)) 
%    title(num2str(j));
%    axis off
%end
%
%for j = 1 : 8
%    figure(4)
%    subplot(2,4,j)
%    imagesc(Conv1.filtres(:,:,j)) 
%    title(num2str(j));
%    axis off
%end