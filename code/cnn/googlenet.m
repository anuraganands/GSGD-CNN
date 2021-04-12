function net = googlenet()
% GOOGLENET A pretrained GoogLeNet convolutional neural network
%
% net = GOOGLENET() returns a pretrained GoogLeNet model that has been trained
% on the ImageNet data set. The output net is a DAGNetwork object.
%
% This function requires the <a href="matlab:helpview(fullfile(docroot,'toolbox','nnet','nnet.map'),'googlenet')">Neural Network Toolbox Model for GoogLeNet Network</a>.
%
% References
% ----------
% - Szegedy, Christian, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, 
%   Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, and Andrew Rabinovich. 
%   "Going deeper with convolutions." In Proceedings of the IEEE conference 
%   on computer vision and pattern recognition, pp. 1-9. 2015.
%
% - https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet
%
% Example - Use GoogLeNet to classify an image
% ------------------------------------------
% % Load a pretrained GoogLeNet CNN
% net = GOOGLENET()
% net.Layers
% plot(net)
%
% % Read the image to classify
% I = imread('peppers.png');
%
% % Crop image to the input size of the network
% sz = net.Layers(1).InputSize
% I = I(1:sz(1), 1:sz(2), 1:sz(3));
%
% % Classify the image using GoogLeNet
% label = classify(net, I)
%
% % Show the image and classification result
% figure
% imshow(I)
% text(10, 20, char(label), 'Color', 'white' )
% 
% See also DAGNetwork, trainNetwork, trainingOptions, alexnet, vgg16, 
% vgg19, importCaffeLayers, importCaffeNetwork

%   Copyright 2017 The MathWorks, Inc.

% Check if support package is installed
breadcrumbFile = 'nnet.internal.cnn.supportpackages.IsGoogLeNetInstalled';
fullPath = which(breadcrumbFile);

if isempty(fullPath)
    name     = 'Neural Network Toolbox Model for GoogLeNet Network';
    basecode = 'GOOGLENET';
    
    error(message('nnet_cnn:supportpackages:NotInstalled', mfilename, name, basecode));
else
    pattern = fullfile(filesep, '+nnet','+internal','+cnn','+supportpackages','IsGoogLeNetInstalled.m');
    idx     = strfind(fullPath, pattern);
    matfile = fullfile(fullPath(1:idx), 'data', 'googlenet.mat');
    data    = load(matfile);
    net     = data.googlenet;
end
