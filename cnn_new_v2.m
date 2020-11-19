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
N=101; # nombre d'image à tester (Attention c'est vite très long)


tab_label=train_labels(offs:N); 
tab_imgr=train_data(offs:N,:,:); # on teste avec N images

tic

num_correct=0;
loss=0;

stepdiv10 = 1;
for i=1:(N-offs)
  label=tab_label(i);
  label=label+1;
  imgr=tab_imgr(i,:,:);
  [d,l,L]=size(imgr);
  img=reshape(imgr,l,L);
  imgn=double((img/255.0))-0.5;
  [out,l,acc]=forward_CNN(Conv1,Pool1,Softmax1,imgn,label);   # un passage forward sur une image
  num_correct=num_correct+acc;
  loss=loss+l;
  
  #fprintf('Indice out Proposition : %d <:::> Label : %d <:::> %d\n', find(out==max(out)),label,acc)
  
  if rem(i,10)==0  #reste de la division par 10
    fprintf('Pour %d images : Average Loss %d | Predictions justes : %d\n',i,loss/(10*stepdiv10),num_correct)
    stepdiv10 = stepdiv10 +1;
  endif
endfor
fprintf("###-------------------------------------------------------###\n")
fprintf("Bilan pour les %d images\n",N-offs)
pourcent_reussite=(num_correct/(N-offs))*100
avgLoss=loss/(N-offs)

toc









