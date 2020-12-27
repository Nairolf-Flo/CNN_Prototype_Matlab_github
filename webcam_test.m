%%fichier pour faire de la reconnaissance avec la webcam
%%Ne marche que sur Matlab avec le package MATLAB Support Package for USB Webcams

cam = webcam(1)
%preview(cam);  %pour voir le retour vidéo
cam.Contrast=0; % 30:la valeur par défaut

while 1==1
    img = snapshot(cam);
    %image(img);
    imgd=imresize(img,[28 28]);
    imgd=rgb2gray(imgd);
    imgd=imcomplement(imgd);
    imgn=double((imgd/255.0))-0.5;
    imagesc(imgn)
    axis off;
    out = forward_convolution(Conv1,imgn);
    figure(3)
    imagesc(out(:,:,2)) %% pour regarder le résultat à travers le filtre 2 de convolution
    axis off
    figure(2)
    out = forward_pooling(Pool1,out);
    out = forward_softmax(Softmax1,out);
    find(out==max(out))-1
    out
    pause(0.5)
  
end



%clear cam