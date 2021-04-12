function net = alexnet()
% ALEXNET A pretrained AlexNet convolutional neural network
% net = ALEXNET() returns a pretrained AlexNet model that was trained
% using the ImageNet data set. The output net is a SeriesNetwork object.
%
% This function requires the <a href="matlab:helpview(fullfile(docroot,'toolbox','nnet','nnet.map'),'alexnet')">Neural Network Toolbox Model for AlexNet Network</a>.
%
% References
% ----------
% - Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. "Imagenet
%   classification with deep convolutional neural networks." Advances in
%   neural information processing systems. 2012.
%
% - https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet
%
% Example - Use AlexNet to classify an image
% ------------------------------------------
% % Load a pretrained AlexNet CNN
% net = ALEXNET()
% net.Layers
%
% % Read the image to classify
% I = imread('peppers.png');
%
% % Crop image to the input size of the network
% sz = net.Layers(1).InputSize
% I = I(1:sz(1), 1:sz(2), 1:sz(3));
%
% % Classify the image using AlexNet
% label = classify(net, I)
%
% % Show the image and classification result
% figure
% imshow(I)
% text(10, 20, char(label), 'Color', 'white' )
% 
% See also SeriesNetwork, trainNetwork, trainingOptions, nnet.cnn.layer,
% VGG16, VGG19.

%   Copyright 2016-2017 The MathWorks, Inc.

% Check if support package is installed
breadcrumbFile = 'nnet.internal.cnn.supportpackages.IsAlexNetInstalled';
fullPath = which(breadcrumbFile);

if isempty(fullPath)
    name     = 'Neural Network Toolbox Model for AlexNet Network';
    basecode = 'ALEXNET';
    
    error(message('nnet_cnn:supportpackages:NotInstalled', mfilename, name, basecode));
else
    pattern = fullfile(filesep, '+nnet','+internal','+cnn','+supportpackages','IsAlexNetInstalled.m');
    idx     = strfind(fullPath, pattern);
    matfile = fullfile(fullPath(1:idx), 'data', 'alexnet.mat');
    data    = load(matfile);
    net     = data.alexnet;
end
