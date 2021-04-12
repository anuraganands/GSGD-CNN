function network = importCaffeNetwork(protofile, datafile, varargin)
% importCaffeNetwork Import Convolutional Neural Network Models from Caffe
%
%  network = importCaffeNetwork(protofile, datafile) imports a pretrained
%  Caffe network as a SeriesNetwork Object
%
%  Inputs:
%  -------
%  protofile - The name of a Caffe PROTOTXT file. The protofile defines the
%              Caffe model architecture. The protofile must be in the current
%              directory, in a directory on the MATLAB path, or include a 
%              full or relative path to a PROTOTXT file.
%
%  datafile -  The name of a Caffe CAFFEMODEL file. The datafile contains the
%              weights of a trained Caffe model. The datafile must be in the 
%              current directory, in a directory on the MATLAB path, or 
%              include a full or relative path to a CAFFEMODEL file.
%
%  [...] = importCaffeNetwork(..., Name, Value) specifies additional
%  name-value pairs described below:
% 
%  'InputSize'   - Size of the input data, specified as a row vector of two
%                  integer numbers corresponding to [height,width] or three
%                  integer numbers corresponding to [height,width,channels].
%              
%                  Default: Uses data from the input layer specified in
%                  the prototxt file if present.  
%
%  'AverageImage' - An 'H-by-W-by-C' matrix that specifies the average image 
%                   to be used for input layer transformation.
%
%                   Default: Uses data from the mean file specified in the
%                            protofile if available.
%
%  'ClassNames'   - A cell array of strings containing the class names
%                   associated with the output layer of the network.
%
%                   Default: If none is provided, the network will be 
%                   initialized with default class names 
%                   {'class1', 'class2', ... }.
%
%  Example - Import a Pretrained Caffe Network to classify an image
%  -----------------------------------------------------------------
%
%  %Import a Caffe Network
%  protofile = 'digitsnet.prototxt';
%  datafile = 'digits_iter_10000.caffemodel';
%  classNames = {'0','1','2','3','4','5','6','7','8','9'};
%  network = importCaffeNetwork(protofile, datafile, 'ClassNames', ...
%            classNames);
%
%  %Read the image to classify
%  digitDatasetPath = fullfile(toolboxdir('nnet'), 'nndemos','nndatasets',...
%                     'DigitDataset');
%  I = imread(fullfile(digitDatasetPath,'5','image4009.png'));
%
%  %Classify the image using the network
%  label = classify(network, I);
%
%  %Show the image and classification result
%  figure
%  imshow(I)
%  title(['Classification result ' char(label)])
%
%  See also importCaffeLayers, trainNetwork, SeriesNetwork, alexnet

% Copyright 2016, The Mathworks Inc.

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
    params = nnet.internal.cnn.caffe.CaffeModelReader.parseImportNetwork(...
        protofile, 'Datafile', datafile, varargin{:});
    
    readerObj = nnet.internal.cnn.caffe.CaffeModelReader(params);
    network = importSeriesNetwork(readerObj);   
end

end

