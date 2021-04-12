%A Strategic Weight Refinement Maneuver for Convolutional Neural Networks - GSGD for Deep Learning Networks

clear all;
clc;
close all;

test_accuracy = [];
validation_accuracy = [];
training_accuracy = []; 
for k=1: 1
     close all;
    % Relative Path of Data Folder
	filepath = 'MNIST_DATASET/MNIST_DATA';

    imdsTrain = imageDatastore(filepath, ...
        'IncludeSubfolders',true, ...
        'LabelSource','foldernames');

     test_filepath = 'MNIST_DATASET/MNIST_DATA_TEST';
     imdsTest = imageDatastore(test_filepath, ...
		'IncludeSubfolders',true, ...
		'LabelSource','foldernames');
   
    %Split Training set into Training and Validation Data
    [imdsTrain,imdsTest]  = splitEachLabel(imdsTrain,0.8,'randomize');
    [imdsTrain,ValidationSet]  = splitEachLabel(imdsTrain,0.8,'randomize');
    
    %Get X and Y Validation Data
    XValidation = ValidationSet;
    YValidation = ValidationSet.Labels;

    %Get the image size
    image = readimage(imdsTrain,1);
    imagesize = size(image)

    % Received from Bayesian
    bestVars.InitialLearnRate = 0.00084469; % 1e-4; 
    bestVars.Momentum = 0.92358 ; %0.97544;
    bestVars.L2Regularization = 3.1505e-07 ; %4.7344e-05
    bestVars.SectionDepth = 1;%2 
    bestVars.filterSz = [3 3]; %[3 3]; 
    bestVars.Rho = 7; %4
    % bestVars.SquaredGradientDecayFactor = 0.98976;

    %Get number of classes
    numClasses = size(imdsTrain.countEachLabel.Count,1);
    imgType = 1; %grayscale
    if size(imagesize,2) == 3
        imgType = 3;
    end

    %Bayesian
    numF = round(0.5*imagesize(1)/sqrt(bestVars.SectionDepth));
    imageSize = [imagesize(1:2) imgType];

    %CNN architecture
    layers = ...
    [
        imageInputLayer(imageSize) % 1 => grayscale images, 3 => RGB
        convBlock(bestVars.filterSz,1*numF) %x4   
        convBlock(bestVars.filterSz,2*numF) %x4   
        convBlock(bestVars.filterSz,4*numF) %x4  
        fullyConnectedLayer(numClasses) % total classes/labels
        softmaxLayer
        classificationLayer
    ];

    %Set Training Options

    % Set 'isGuided' parameters to true for GSGD and supply 'Rho', 'RevisitBatchNum' and
    % 'VerificationSetNum' values
    %Simply remove the above parameters to run without GSGD or set 'isGuided'
    %to false

    %'Rho' - number of iterations  to run for collection and checking of consistent data
    %        before guided approach is activated to update the weights with consistent
    %        data

    %'RevisitBatchNum' - number of previous batches to revisit and 
    %                   check how it performs on present batch weights

    %'VerificationSetNum' - number of batches to set aside at the beginning of each epoch. 
    %                       Each batch gets picked randomly from this set to attain true
    %                       error on weights updated by each batch during
    %                       training

    options = trainingOptions('sgdm', ...
        'Momentum',bestVars.Momentum, ...
        'MaxEpochs',45,...
        'ExecutionEnvironment','gpu', ...
        'MiniBatchSize',256, ...
        'InitialLearnRate',bestVars.InitialLearnRate, ...
        'Verbose',false, ...
        'L2Regularization',bestVars.L2Regularization, ...
        'ValidationData',{XValidation,YValidation}, ...
        'ValidationPatience', inf, ...
        'ValidationFrequency', 10, ...
        'isGuided', true, ...
        'Rho',  bestVars.Rho, ...
        'RevisitBatchNum', 2, ...
        'VerificationSetNum', 4, ...
        'Plots','training-progress');

    warning off parallel:gpu:devie:DeviceLibsNeedsRecompiling
    try
        gpuArray.eye(2)^2;
    catch ME
    end
    try
        nnet.internal.cnngpu.reluForward(1);
    catch ME
    end


    % Train the network 
    [net,info] = trainNetwork(imdsTrain,layers,options);

	% collect information from info

    %Get Training Accuracy
    YPred = classify(net,imdsTest);
    YTest = imdsTest.Labels;
    acc = (sum(YPred == YTest)/numel(YTest)) * 100;
    test_accuracy = [test_accuracy acc];

    fprintf('Highest training SR: %.2f\n',max(info.TrainingAccuracy));
    training_accuracy = [training_accuracy info.TrainingAccuracy];
    
    %Avoid overfitting
    validation_accuracy = [validation_accuracy info.ValidationAccuracy];
    fprintf('Highest validation SR: %.2f\n',max(info.ValidationAccuracy));
    fprintf("%i Test Accuracy is: %i\n",k, acc);
end
fprintf('test_accuracy AT THE END OF FOR LOOP: %.2f\n',test_accuracy);
%Save the Model. If model maybe needed in future, then rename it otherwise
%it will be overwritten after every training
% save('Models/trainedmodel.mat','net');

end
