%categories = {'000', '001', '002', '010', '011', '012', '022', '100', '101', '102', '110', '111', '112', '120', '121', '122', '200', '201', '202', '210', '211', '212', '220', '221', '222'}
folderName = 'C:\Users\Jesper\Documents\GitHub\Matlab-ML\imagedata';
lbls = importdata('labels.txt');
lbls = num2str(lbls(:, :))



labels = [];
for l = 1:size(lbls, 1)
    labels = [labels; strrep(lbls(l, :), " ", "")];
end

labels = categorical(labels);

imds = imageDatastore(folderName, 'ReadFcn', @customreader);
imds.Labels = labels;



trainingNumFiles = 20;
rng(1) % For reproducibility
[trainDigitData,testDigitData] = splitEachLabel(imds,...
    trainingNumFiles,'randomize');
         

conv1 = convolution2dLayer(5,20,'Padding',0,...
                     'BiasLearnRateFactor',2,...
                     'name','conv1');
conv1.Weights = single(randn([5 5 1 20])*0.0001);
conv1.Bias = single(randn([1 1 20])*0.00001+1);

layers = [imageInputLayer([225 301 1]);
          conv1;
          reluLayer();
          maxPooling2dLayer(2,'Stride',2);
          fullyConnectedLayer(27);
          softmaxLayer();
          classificationLayer()];
      
options = trainingOptions('sgdm','MaxEpochs',20,...
    'InitialLearnRate',0.001);
    
convnet = trainNetwork(trainDigitData, layers, options);

YTest = classify(convnet,testDigitData);
TTest = testDigitData.Labels;
accuracy = sum(YTest == TTest)/numel(TTest);
%net = SeriesNetwork();






