###############################################################
## Script pour tester le réseau CNN sur un ensemble d'images ##
###############################################################

pkg load nan
[train_data, test_data, train_labels, test_labels] = load_mnist();# un peu lourd, charge la base de digits

##    img1=imread('im_test.jpg','jpg');  ##image de test
##    img2=imread('ensil.jpg','jpg');  ##image de test
##    img3=imread('ensilr2.jpg','jpg');  ##image de test

#images à tester :
offs=1;
N=1001; # nombre d'image à tester (Attention c'est vite très long)
learn_rate=0.005;

## Initialisation des différentes couches du réseau CNN
Conv1 = Conv3x3(8);             # Couche de convolution à 8 filtres 28x28x1 -> 26x26x8
Pool1 = MaxPool2();             # Couche de Maxpooling              28x28x1 -> 26x26x8
Softmax1 = Softmax(13*13*8,10); # Couche de Softmax                 13x13x8 -> 10
 
tab_label=train_labels(offs:N); 
tab_imgr=train_data(offs:N,:,:); # on teste avec N images

tic

num_correct=0;
loss=0;

%debug biais
m=1:(N-offs);
%p=zeros(N-offs,10);
%q=zeros(N-offs,50);
r=zeros(N-offs,3,3);


stepdiv10 = 1;
for i=1:(N-offs)
  label=tab_label(i);
  label=label+1;
  imgr=tab_imgr(i,:,:);
  [d,l,L]=size(imgr);
  img=reshape(imgr,l,L);
  imgn=double((img/255.0))-0.5;
  [l,acc]=train_CNN(Conv1,Pool1,Softmax1,imgn,label,learn_rate);   # un passage forward sur une image
  num_correct=num_correct+acc;
  loss=loss+l;
  
%  b=Softmax1.dbiais;
%  w=Softmax1.dweights(501:550,6);
%  p(i,:)=b;
%  q(i,:)=w;
  
  r(i,:,:)=Conv1.filtres(:,:,4);
 
  
  #fprintf('Indice out Proposition : %d <:::> Label : %d <:::> %d\n', find(out==max(out)),label,acc)
  
##  if rem(i,100)==0  #reste de la division par 10
##    fprintf('Pour 10 images : Average Loss %d | Predictions justes : %d\n',loss/(10*stepdiv10),num_correct)
##    stepdiv10 = stepdiv10 +1;
##  endif

  if rem(i,100)==0  #reste de la division par 10 
    fprintf('Step %d : Past 100 steps: Average Loss %d | Accuracy : %d\n',i, loss/100,num_correct)
    num_correct=0;
    loss=0;
  endif
endfor


%fprintf("###-------------------------------------------------------###\n")
%fprintf("Bilan pour les %d images\n",N-offs)
%pourcent_reussite=(num_correct/(N-offs))*100
%avgLoss=loss/(N-offs)

toc

%figure(1)
%title("biais")
%plot(m,p)
%figure(2)
%title("weights")
%plot(m,q)

figure(3)
title ("filtres")
r=reshape(r,N-offs,9);
plot(m,r)









