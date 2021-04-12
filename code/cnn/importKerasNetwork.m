function Network = importKerasNetwork(ModelFile, varargin)
%importKerasNetwork  Import a Keras network and weights from a model file
%or from separate architecture and weight files.
% 
% Network = importKerasNetwork(ModelFile) imports a pretrained Keras
% network and its weights from ModelFile.
%
%  Inputs:
%  -------
%
%  ModelFile       - Name of a Keras file, which may be a string or a
%                    char array. The file may be in HDF5 (.h5) or JSON
%                    (.json) format. If it is in JSON format, a weight file
%                    must also be passed using the 'WeightFile' argument.
%
%  [...] = importKerasNetwork(..., Name, Value) specifies additional
%  name-value pairs described below:
% 
%  'WeightFile'    - Name of the HDF5 from which to import weights when
%                    ModelFile does not include the weights.
% 
%  'OutputLayerType' - Either 'classification' or 'regression'. Specifies
%                    the type of output layer that will be appended to the
%                    imported network when ModelFile does not specify a
%                    loss function.
% 
%  'ImageInputSize' - A vector specifying the size of the input images for
%                     the network, used when the model file does not
%                     specify the size. It must be a row vector of two or
%                     three numbers: [height, width] or [height, width,
%                     channels].
% 
%  'ClassNames'    - A string array or cell array of character vectors
%                    specifying the class names of the final classification
%                    output layer. Only valid for classification networks.
%                    Default: string(1:N) for N classes.
%
%  Outputs:
%  -------
%
%  Network         - A SeriesNetwork if the Keras network is of type
%                    'Sequential', or a DAGNetwork if the Keras network is
%                    of type 'Model'.
%
%
% Example - Import a pretrained Keras network to classify an image
% -----------------------------------------------------------------
% 
% %Import a Keras Network
% netfile = 'digitsDAGnet.h5';
% classNames = {'0','1','2','3','4','5','6','7','8','9'};
% network = importKerasNetwork(netfile, 'ClassNames', classNames);
% 
% %Read the image to classify
% digitDatasetPath = fullfile(toolboxdir('nnet'), 'nndemos','nndatasets',...
%                   'DigitDataset');
% I = imread(fullfile(digitDatasetPath,'5','image4009.png'));
% 
% %Classify the image using the network
% label = classify(network, I);
% 
% %Show the image and classification result
% figure
% imshow(I)
% title(['Classification result ' char(label)])
%
%  See also importKerasLayers

% Copyright 2017 The Mathworks, Inc.

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
    Network = nnet.internal.cnn.keras.importKerasNetwork(ModelFile, varargin{:});
catch ME
    throw(ME);
end
end
