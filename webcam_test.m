cam = webcam(1)
preview(cam);
label=1;

while 1==1
    img = snapshot(cam);
    %image(img);
    imgd=imresize(img,[28 28]);
    imgd=rgb2gray(imgd);
    imgd=imcomplement(imgd)
    imgn=double((imgd/255.0))-0.5;
    imagesc(imgn)
    axis off;
  [out,l,acc]=forward_CNN(Conv1,Pool1,Softmax1,imgn,label);
  find(out==max(out))
  out
  pause(0.5)
  
end



%clear cam