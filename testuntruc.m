% fichier devenu inutile

mnistfilenames = cell(4,1);
mnistfilenames{1} = "train-images-idx3-ubyte";
mnistfilenames{2} = "train-labels-idx1-ubyte";
mnistfilenames{3} = "t10k-images-idx3-ubyte";
mnistfilenames{4} = "t10k-labels-idx1-ubyte";
[TrainImages TrainLabels TestImages TestLabels]=mnistread(mnistfilenames);
% load data for training and testing from files
load mnistbbdbn;
    % load the trained DNN
    dbn = bbdbn;
    % set dbn as the trained net
    N = 10;
    %number of test data to analyze
    IN = TestImages(1:N,:);
    %load only N records of testing data
    OUT = TestLabels(1:N,:);
    %load only N records of testing data
    for i=1:N
      imshow(reshape(IN(i,:),28,28));
      name = ["print img-training-",num2str(i),".jpg -djpeg"]
      eval(name);
      %save the plot, file in jpeg format
    end
    
  % v2h: get the output of the DNN
  out = v2h( dbn, IN );
  % get the maximum values (m) of out and indexes (ind)
  [m ind] = max(out,[],2);
  out = zeros(size(out));
  % initialize the variable out
  % Now, fill with ones where the maximum values where located (ind):
  for i=1:size(out,1)
    out(i,ind(i)) = 1;
  end
  % Now compare out vs OUT. Let say the output of the DNN (out) vs
  % the desired output (OUT)
  ErrorRate = abs(OUT-out);
  % analytically compare OUT vs out
  % sum(ErrorRate,2) performs the sum in two dimensions,
  % first by row then the resulting column.
  % It is divided by 2; if some output fails, the sum will count twice.
  % mean gives us the percentage of error, known as error rate.
  ErrorRate = mean(sum(ErrorRate,2)/2) 
  % finally the Error rate is obtained