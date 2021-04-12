function net = resnet101()
% RESNET101 A pretrained ResNet-101 convolutional neural network
%
% net = RESNET101() returns a pretrained ResNet-101 model that has been trained
% on the ImageNet data set. The output net is a DAGNetwork object.
%
% This function requires the <a href="matlab:helpview(fullfile(docroot,'toolbox','nnet','nnet.map'),'resnet101')">Neural Network Toolbox Model for ResNet-101 Network</a>.
%
% References
% ----------
% - He, Kaiming, Zhang, Xiangyu, Ren, Shaoqing, Sun, Jian. 
%   "Deep Residual Learning for Image Recognition." In Proceedings of the 
%   IEEE conference on computer vision and pattern recognition, pp. 770-778. 2016.
%
% - https://github.com/KaimingHe/deep-residual-networks
%
% Example - Use ResNet-101 to classify an image
% ------------------------------------------
% % Load a pretrained ResNet-101 CNN
% net = RESNET101()
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
% % Classify the image using ResNet-101
% label = classify(net, I)
%
% % Show the image and classification result
% figure
% imshow(I)
% text(10, 20, char(label), 'Color', 'white' )

% 
% See also DAGNetwork, trainNetwork, trainingOptions, alexnet, vgg16, 
% vgg19, googlenet, resnet50, inceptionv3, importCaffeLayers, importCaffeNetwork

%   Copyright 2017 The MathWorks, Inc.

% Check if support package is installed
breadcrumbFile = 'nnet.internal.cnn.supportpackages.IsResNet101Installed';
fullPath = which(breadcrumbFile);

if isempty(fullPath)
    name     = 'Neural Network Toolbox Model for ResNet-101 Network';
    basecode = 'RESNET101';
    
    error(message('nnet_cnn:supportpackages:NotInstalled', mfilename, name, basecode));
else
    pattern = fullfile(filesep, '+nnet','+internal','+cnn','+supportpackages','IsResNet101Installed.m');
    idx     = strfind(fullPath, pattern);
    matfile = fullfile(fullPath(1:idx), 'data', 'resnet101.mat');
    data    = load(matfile);
    net     = data.resnet101;
end
