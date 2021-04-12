function net = inceptionv3()
% INCEPTIONV3 A pretrained Inception-v3 convolutional neural network
%
% net = INCEPTIONV3() returns a pretrained Inception-v3 model that has been trained
% on the ImageNet data set. The output net is a DAGNetwork object.
%
% This function requires the <a href="matlab:helpview(fullfile(docroot,'toolbox','nnet','nnet.map'),'inceptionv3')">Neural Network Toolbox Model for Inception-v3 Network</a>.
%
% References
% ----------
% - Szegedy, Christian, Vanhoucke, Vincent, Ioffe, Sergey, Shlens, Jonathon & Wojna, Zbigniew. 
%   "Rethinking the Inception Architecture for Computer Vision." In Proceedings of the 
%   IEEE conference on computer vision and pattern recognition, pp. 2818-2826. 2016.
%
% - https://keras.io/applications/#inceptionv3
%
% Example - Use Inception-v3 to classify an image
% ------------------------------------------
% % Load a pretrained Inception-v3 CNN
% net = INCEPTIONV3()
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
% % Classify the image using Inception-v3
% label = classify(net, I)
%
% % Show the image and classification result
% figure
% imshow(I)
% text(10, 20, char(label), 'Color', 'white' )

% 
% See also DAGNetwork, trainNetwork, trainingOptions, alexnet, vgg16, 
% vgg19, googlenet, resnet50, importKerasLayers, importKerasNetwork

%   Copyright 2017 The MathWorks, Inc.

% Check if support package is installed
breadcrumbFile = 'nnet.internal.cnn.supportpackages.IsInceptionV3Installed';
fullPath = which(breadcrumbFile);

if isempty(fullPath)
    name     = 'Neural Network Toolbox Model for Inception-v3 Network';
    basecode = 'INCEPTIONV3';
    
    error(message('nnet_cnn:supportpackages:NotInstalled', mfilename, name, basecode));
else
    pattern = fullfile(filesep, '+nnet','+internal','+cnn','+supportpackages','IsInceptionV3Installed.m');
    idx     = strfind(fullPath, pattern);
    matfile = fullfile(fullPath(1:idx), 'data', 'inceptionv3.mat');
    data    = load(matfile);
    net     = data.inceptionv3;
end
