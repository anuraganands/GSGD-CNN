function layers = importCaffeLayers(protofile, varargin)
% importCaffeLayers Import Convolutional Neural Network Layers from Caffe
%
%  layers = importCaffeLayers(protofile) imports the Caffe model
%  architecture defined in a PROTOTXT file named protofile. The model
%  architecture is returned in layers, an array of Layer objects. protofile
%  must be in the current directory, in a directory on the MATLAB path or
%  include a full or relative path to a prototxt file.
%
%  [...] = importCaffeLayers(..., Name, Value) specifies additional
%  name-value pairs described below:
% 
%  'InputSize'   - Size of the input data, specified as a row vector of two
%                  integer numbers corresponding to [height,width] or three
%                  integer numbers corresponding to [height,width,channels].
%              
%                  Default: Uses data from the input layer specified in
%                  the prototxt file if present.              
%
%  Example - Import Caffe Network Layers And Train a Network
%  ---------------------------------------------------------
%
%  %Import Layers from Caffe Network
%  layers = importCaffeLayers('digitsnet.prototxt');
%
%  %Load a dataset for training a classifier to recognize the digits
%  digitDatasetPath = fullfile(toolboxdir('nnet'), 'nndemos', ...
%                     'nndatasets', 'DigitDataset');
%  digitData = imageDatastore(digitDatasetPath, ...
%      'IncludeSubfolders',true,'LabelSource','foldernames');
%
%  %Partition the dataset into training & test images
%  rng(1) % For reproducibility
%  trainingFileSplitRatio = 0.6;
%  [trainDigitData,testDigitData] = splitEachLabel(digitData,...
%      trainingFileSplitRatio,'randomize');
%
%  %Set some training options
%  options = trainingOptions('sgdm',...
%      'Plots', 'training-progress',...
%      'MaxEpochs',20,...
%      'InitialLearnRate',0.001);
%
%  %Train network
%  convnet = trainNetwork(trainDigitData,layers,options);
%
%  %Read image for classification
%  I = imread(fullfile(digitDatasetPath,'5','image4009.png'));
%  figure
%  imshow(I)
%
%  %Classify the image using the network
%  label = classify(convnet, I);
%  title(['Classification result ' char(label)])
%
%  See also importCaffeNetwork, trainNetwork, SeriesNetwork, alexnet

% Copyright 2016-2017 The MathWorks, Inc.

% Check if spkg is installed
breadcrumbFile = 'nnet.internal.cnn.supportpackages.isCaffeImporterInstalled';
fullpath = which(breadcrumbFile);

if isempty(fullpath)
    % Not installed; throw an error
    name = 'Neural Network Toolbox Importer for Caffe Models';
    basecode = 'CAFFEIMPORTER';
    error(message('nnet_cnn:supportpackages:NotInstalled', ...
        mfilename, name, basecode))
else
    params =  nnet.internal.cnn.caffe.CaffeModelReader.parseImportLayers(protofile, varargin{:});
    readerObj = nnet.internal.cnn.caffe.CaffeModelReader(params);
    layers = importLayers(readerObj);
end

end