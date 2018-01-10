rootFolder = fullfile('C:\Users\Jesper\Documents\GitHub\Matlab-ML\data', '101_ObjectCategories');
categories = {'snoopy', 'electric_guitar', 'pizza'};

imds = imageDatastore(fullfile(rootFolder, categories), 'LabelSource', 'foldernames');

labeling = countEachLabel(imds);

minSetCount = min(labeling{:, 2});
imds = splitEachLabel(imds, minSetCount, 'randomize');
labeling = countEachLabel(imds);

net = alexnet();