function Layers = importKerasLayers(ModelFile, varargin)
%importKerasLayers  Import a Keras network architecture from a model file.
% 
% Layers = importKerasLayers(ModelFile) imports a Keras network
% architecture from ModelFile.
%
%  Inputs:
%  -------
%
%  ModelFile       - Name of a Keras model file, specified as a string
%                    or a char array. The file must be HDF5 (.h5) or JSON
%                    (.json) format.
%
%  Layers = importKerasLayers(ModelFile, 'PARAM1', VAL1, 'PARAM2', VAL2, ...)
%  specifies additional name-value pairs described below:
% 
%  'OutputLayerType' - Either 'classification' or 'regression'. Specifies
%                    the type of output layer that will be appended to the
%                    imported layers when ModelFile does not specify a loss
%                    function.
% 
%  'ImageInputSize' - A vector specifying the size of the input images for
%                     the network, used when the model file does not
%                     specify the size. It must be a row vector of two or
%                     three numbers: [height, width] or [height, width,
%                     channels].
% 
%  'ImportWeights' - Logical scalar indicating whether weights should
%                    also be imported. Possible values are:
%                      - true - import the weights from ModelFile.
%                        ModelFile must have HDF5 format. If ModelFile does
%                        not include the weights, then you must specify a
%                        separate file that includes weights, using the
%                        'WeightFile' name-value pair argument. 
%                      - false (default) - do not import weights
% 
%  'WeightFile'    - Name of the HDF5 file from which to import weights
%                    when ModelFile does not include the weights.  To use
%                    this name-value pair argument, you must also set
%                    'ImportWeights' to true.
%
%  Outputs:
%  -------
%
%  Layers          - A Layer array if the Keras network is of type
%                    'Sequential', or a LayerGraph if the Keras network is
%                    of type 'Model'.
%
%
%  Example - Import Keras Network Layers And Train a Network
%  ---------------------------------------------------------
%
%  %Import Layers from Keras Network
%  layers = importKerasLayers('digitsDAGnet.h5');
%
%  %Load a dataset for training a classifier to recognize the digits
%  digitDatasetPath = fullfile(toolboxdir('nnet'), 'nndemos', ...
%                     'nndatasets', 'DigitDataset');
%  digitData = imageDatastore(digitDatasetPath, ...
%      'IncludeSubfolders',true,'LabelSource','foldernames');
%  %Partition the dataset into training & test images
%  rng(1) % For reproducibility
%  trainingFileSplitRatio = 0.6;
%  [trainDigitData,testDigitData] = splitEachLabel(digitData,...
%      trainingFileSplitRatio,'randomize');
%
%  %Set some training options
%  options = trainingOptions('sgdm',...
%      'MaxEpochs',20,...
%      'Plots','training-progress',...
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
%  See also findPlaceholderLayers, trainNetwork, importKerasNetwork

% Copyright 2017 The MathWorks, Inc.

%% Check if support package is installed
breadcrumbFile = 'nnet.internal.cnn.supportpackages.isKerasImporterInstalled';
fullpath = which(breadcrumbFile);
if isempty(fullpath)
    % Not installed; throw an error
    name = 'Neural Network Toolbox Importer for Keras Models';
    basecode = 'KERASIMPORTER';
    error(message('nnet_cnn:supportpackages:NotInstalled', ...
        mfilename, name, basecode));
end

%% Call the main function
try
    Layers = nnet.internal.cnn.keras.importKerasLayers(ModelFile, varargin{:});
catch ME
    throw(ME);
end
end
