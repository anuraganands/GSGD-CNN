% Neural Network Toolbox convolutional neural network (CNN) functions.
%
% Training a CNN
%   augmentedImageDatastore        - Generate batches of augmented image data
%   imageDataAugmenter             - Configure image data augmentation
%   trainNetwork                   - Train a neural network
%   trainingOptions                - Options for training a neural network
%
% Layers of a CNN
%   additionLayer                  - Addition layer
%   averagePooling2dLayer          - Average pooling layer
%   batchNormalizationLayer        - Batch normalization layer
%   bilstmLayer                    - Bidirectional long short-term memory (biLSTM) layer
%   classificationLayer            - Classification output layer for a neural network
%   clippedReluLayer               - Clipped rectified linear unit (ReLU) layer
%   convolution2dLayer             - 2-D convolution layer for Convolutional Neural Networks
%   crossChannelNormalizationLayer - Local response normalization along channels
%   depthConcatenationLayer        - Depth concatenation layer
%   dropoutLayer                   - Dropout layer
%   fullyConnectedLayer            - Fully connected layer
%   imageInputLayer                - Image input layer
%   leakyReluLayer                 - Leaky rectified linear unit (ReLU) layer
%   lstmLayer                      - Long short-term memory (LSTM) layer
%   maxPooling2dLayer              - Max pooling layer
%   maxUnpooling2dLayer            - Max unpooling layer
%   regressionLayer                - Regression output layer for a neural network
%   reluLayer                      - Rectified linear unit (ReLU) layer
%   sequenceInputLayer             - Sequence input layer
%   softmaxLayer                   - Softmax layer
%   transposedConv2dLayer          - 2-D transposed convolution layer
%
% Define New Layers for Deep Learning
%   checkLayer                     - Check layer validity
%   nnet.layer.ClassificationLayer - Interface for classification layers
%   nnet.layer.Layer               - Interface for custom layers
%   nnet.layer.RegressionLayer     - Interface for regression layers
%
% Extract and Visualize Features, Predict Outcomes
%   deepDreamImage                 - Visualize network features using Deep Dream
%   SeriesNetwork.classify         - Classify data with a network
%   SeriesNetwork.predict          - Make predictions on data with a network
%   SeriesNetwork.activations      - Computes network layer activations
%   SeriesNetwork.predictAndUpdateState     - Make predictions and update network state
%   SeriesNetwork.classifyAndUpdateState    - Classify and update network state
%   SeriesNetwork.resetState       - Reset network state
%   SeriesNetwork                  - Network with layers arranged in series
%   DAGNetwork                     - Network with layers arranged as a directed acyclic graph
%   layerGraph                     - Create a layer graph
%
% Using Pretrained Networks
%   alexnet                        - Pretrained AlexNet convolutional neural network
%   googlenet                      - Pretrained GoogLeNet convolutional neural network
%   inceptionv3                    - Pretrained Inception-v3 convolutional neural network
%   resnet50                       - Pretrained ResNet-50 convolutional neural network
%   resnet101                      - Pretrained ResNet-101 convolutional neural network
%   vgg16                          - Pretrained VGG-16 convolutional neural network
%   vgg19                          - Pretrained VGG-19 convolutional neural network
%   importCaffeLayers              - Import Convolutional Neural Network Layers from Caffe
%   importCaffeNetwork             - Import Convolutional Neural Network Models from Caffe
%   importKerasLayers              - Import Convolutional Neural Network Layers from Keras
%   importKerasNetwork             - Import Convolutional Neural Network Models from Keras

% Copyright 2015-2018 The MathWorks, Inc.
