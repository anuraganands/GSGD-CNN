function net = vgg16()
%VGG16 Pretrained VGG-16 convolutional neural network
%
%   net = VGG16() returns a pretrained VGG-16 SeriesNetwork object. The
%   convolutional neural network has been trained on the ImageNet data set.
%
%   This function requires the <a href="matlab:helpview(fullfile(docroot,'toolbox','nnet','nnet.map'),'vgg16')">Neural Network Toolbox Model for VGG-16 Network</a>.
%
%   References
%   ----------
%   [1] Simonyan, Karen, and Andrew Zisserman. "Very deep convolutional
%       networks for large-scale image recognition." arXiv preprint
%       arXiv:1409.1556 (2014).
%
%   [2] http://www.robots.ox.ac.uk/~vgg/research/very_deep/
%
%   Example: Use VGG-16 to classify an image
%   ----------------------------------------
%     % Load a pretrained VGG-16 convolutional neural network (CNN)
%     net = vgg16;
%
%     % The network is made of 16 convolutional and fully connected layers
%     net.Layers
%
%     % Read the image to classify
%     I = imread('peppers.png');
%
%     % Crop the image to the input size of the network
%     sz = net.Layers(1).InputSize;
%     I = I(1:sz(1), 1:sz(2), 1:sz(3));
%
%     % Classify the image using VGG-16
%     label = classify(net, I)
%
%     % Show the image and classification result
%     figure
%     imshow(I)
%     text(10, 20, char(label), 'Color', 'white')
%
%   See also ALEXNET, SeriesNetwork, VGG19.

%   Copyright 2016-2017 The MathWorks, Inc.

% Check if spkg is installed.
% Note: the network was created using importCaffeNetwork on the files
% provided in Ref [2] above.
breadcrumbFile = 'nnet.internal.cnn.supportpackages.IsVGG16Installed';
fullpath = which(breadcrumbFile);

if isempty(fullpath)
    % Not installed; throw an error
    name = 'Neural Network Toolbox Model for VGG-16 Network';
    basecode = 'VGG16';
    error(message('nnet_cnn:supportpackages:NotInstalled', ...
        mfilename, name, basecode))
else
    matfile = nnet.internal.cnn.supportpackages.getVGG16DataLocation(fullpath);
    data    = load(matfile);
    net     = data.vgg16;
end
