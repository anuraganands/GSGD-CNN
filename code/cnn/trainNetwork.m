function [trainedNet, info] = trainNetwork(varargin)
% trainNetwork   Train a neural network
%
%   trainedNet = trainNetwork(ds, layers, options) trains and returns a
%   network trainedNet for a classification problem. ds is an
%   imageDatastore with categorical labels or a MiniBatchable Datastore
%   with responses, layers is an array of network layers or a LayerGraph
%   and options is a set of training options.
%
%   trainedNet = trainNetwork(X, Y, layers, options) trains and returns a
%   network, trainedNet. The format for X depends on the input layer. For
%   an image input layer, X is a numeric array of images arranged so that
%   the first three dimensions are the width, height and channels, and the
%   last dimension indexes the individual images. In a classification
%   problem, Y specifies the labels for the images as a categorical vector.
%   In a regression problem, Y contains the responses arranged as a matrix
%   of size number of observations by number of responses, or a four
%   dimensional numeric array, where the last dimension corresponds to the
%   number of observations. 
%
%   trainedNet = trainNetwork(C, Y, layers, options) trains an LSTM network
%   for classifcation and regression problems for sequence or time-series
%   data. layers must define an LSTM network. It must begin with a sequence
%   input layer. C is a cell array containing sequence or time-series
%   predictors. The entries of C are D-by-S matrices where D is the number
%   of values per timestep, and S is the length of the sequence. For
%   sequence-to-label classification problems, Y is a categorical vector of
%   labels. For sequence-to-sequence classification problems, Y is a cell
%   array of categorical sequences. For sequence-to-one regression
%   problems, Y is a matrix of targets. For sequence-to-sequence regression
%   problems, Y is a cell array of numeric sequences. For
%   sequence-to-sequence problems, the number of time steps of the
%   sequences in Y must be identical to the corresponding predictor
%   sequences in C. For sequence-to-sequence problems with one observation,
%   C can be a matrix, and Y must be a categorical sequence of labels or a
%   matrix of responses.
%
%   trainedNet = trainNetwork(tbl, layers, options) trains and returns a
%   network, trainedNet. For networks with an image input layer, tbl is a
%   table containing predictors in the first column as either absolute or
%   relative image paths or images. Responses must be in the second column
%   as categorical labels for the images. In a regression problem,
%   responses must be in the second column as either vectors or cell arrays
%   containing 3-D arrays or in multiple columns as scalars. For networks
%   with a sequence input layer, tbl is a table containing absolute or
%   relative MAT file paths of predictors in the first column. For a
%   sequence-to-label classification problem, the second column must be a
%   categorical vector of labels. For a sequence-to-one regression problem,
%   the second column must be a numeric array of responses or in multiple
%   columns as scalars. For a sequence-to-sequence classification problem,
%   the second column must be an absolute or relative file path to a MAT
%   file with a categorical sequence. For a sequence-to-sequence regression
%   problem, the second column must be an absolute or relative file path to
%   a MAT file with a numeric response sequence.
%
%   trainedNet = trainNetwork(tbl, responseName, ...) trains and returns a
%   network, trainedNet. responseName is a character vector specifying the
%   name of the variable in tbl that contains the responses.
%
%   trainedNet = trainNetwork(tbl, responseNames, ...) trains and returns a
%   network, trainedNet, for regression problems. responseNames is a cell
%   array of character vectors specifying the names of the variables in tbl
%   that contain the responses.
%
%   [trainedNet, info] = trainNetwork(...) trains and returns a network,
%   trainedNet. info contains information on training progress.
%
%   Example 1:
%       Train a convolutional neural network on some synthetic images
%       of handwritten digits. Then run the trained network on a test
%       set, and calculate the accuracy.
%
%       [XTrain, YTrain] = digitTrain4DArrayData;
%
%       layers = [ ...
%           imageInputLayer([28 28 1])
%           convolution2dLayer(5,20)
%           reluLayer
%           maxPooling2dLayer(2,'Stride',2)
%           fullyConnectedLayer(10)
%           softmaxLayer
%           classificationLayer];
%       options = trainingOptions('sgdm', 'Plots', 'training-progress');
%       net = trainNetwork(XTrain, YTrain, layers, options);
%
%       [XTest, YTest] = digitTest4DArrayData;
%
%       YPred = classify(net, XTest);
%       accuracy = sum(YTest == YPred)/numel(YTest)
%
%   Example 2:
%       Train a long short-term memory network to classify speakers of a
%       spoken vowel sounds on preprocessed speech data. Then make
%       predictions using a test set, and calculate the accuracy.
%
%       [XTrain, YTrain] = japaneseVowelsTrainData;
%
%       layers = [ ...
%           sequenceInputLayer(12)
%           lstmLayer(100, 'OutputMode', 'last')
%           fullyConnectedLayer(9)
%           softmaxLayer
%           classificationLayer];
%       options = trainingOptions('adam', 'Plots', 'training-progress');
%       net = trainNetwork(XTrain, YTrain, layers, options);
%
%       [XTest, YTest] = japaneseVowelsTestData;
%
%       YPred = classify(net, XTest);
%       accuracy = sum(YTest == YPred)/numel(YTest)
%
%   Example 3:
%       Train a network on synthetic digit data, and measure its
%       accuracy:
%
%       [XTrain, YTrain] = digitTrain4DArrayData;
%
%       layers = [
%           imageInputLayer([28 28 1], 'Name', 'input')
%           convolution2dLayer(5, 20, 'Name', 'conv_1')
%           reluLayer('Name', 'relu_1')
%           convolution2dLayer(3, 20, 'Padding', 1, 'Name', 'conv_2')
%           reluLayer('Name', 'relu_2')
%           convolution2dLayer(3, 20, 'Padding', 1, 'Name', 'conv_3')
%           reluLayer('Name', 'relu_3')
%           additionLayer(2,'Name', 'add')
%           fullyConnectedLayer(10, 'Name', 'fc')
%           softmaxLayer('Name', 'softmax')
%           classificationLayer('Name', 'classoutput')];
%
%       lgraph = layerGraph(layers);
%
%       lgraph = connectLayers(lgraph, 'relu_1', 'add/in2');
%
%       plot(lgraph);
%
%       options = trainingOptions('sgdm', 'Plots', 'training-progress');
%       [net,info] = trainNetwork(XTrain, YTrain, lgraph, options);
%
%       [XTest, YTest] = digitTest4DArrayData;
%       YPred = classify(net, XTest);
%       accuracy = sum(YTest == YPred)/numel(YTest)
%
%   See also nnet.cnn.layer, trainingOptions, SeriesNetwork, DAGNetwork, LayerGraph.

%   Copyright 2015-2018 The MathWorks, Inc.

narginchk(3,4);

try
    [layersOrGraph, opts, X, Y] = iParseInputArguments(varargin{:});
    [trainedNet, info] = doTrainNetwork(layersOrGraph, opts, X, Y);
catch e
    iThrowCNNException( e );
end

end

function [trainedNet, info] = doTrainNetwork(layersOrGraph, opts, X, Y)

haveDAGNetwork = iHaveDAGNetwork(layersOrGraph);

analysis = iInferParameters(layersOrGraph);
layersGraph = analysis.LayerGraph;
layers = analysis.ExternalLayers;
internalLayers = analysis.InternalLayers;
isRNN = nnet.internal.cnn.util.isRNN( internalLayers );

% Create an internal to external layers map
layersMap = iLayersMap( layers );

% Validate options
iValidateOptions( opts );

% Validate training data
iValidateTrainingDataForProblem( X, Y, layers );

% Set desired precision
precision = nnet.internal.cnn.util.Precision('single');

% Set up and validate parallel training
executionSettings = iSetupExecutionEnvironment( opts, isRNN, X );

% Create a training dispatcher
trainingDispatcher = iCreateTrainingDataDispatcher(X, Y, opts, precision, ...
    executionSettings, layersMap.externalLayers( internalLayers ));

% Assert that the input data has a valid size for the network in use and
% the response size matches the output of the network. Rethrow exceptions
% as if they were thrown from the main function
if haveDAGNetwork
    iValidateTrainingDataSizeForDAGNetwork(trainingDispatcher, layersGraph);
else
    iValidateTrainingDataSizeForNetwork(trainingDispatcher, internalLayers);
end

% Initialize learnable parameters
internalLayers = iInitializeParameters(internalLayers, precision);

% Store labels into cross entropy layer or response names into mean-squared
% error layer
if iIsClassificationNetwork( internalLayers )
    internalLayers = iMaybeStoreCategories(internalLayers, trainingDispatcher);
else
    responseNames = trainingDispatcher.ResponseNames;
    internalLayers = iStoreResponseNames(internalLayers, responseNames);
end

% Create the network
trainedNet = iCreateInternalNetwork( layersGraph, internalLayers, haveDAGNetwork );

% Convert learnable parameters to the correct format
trainedNet = trainedNet.prepareNetworkForTraining( executionSettings );

% Create a validation dispatcher if validation data was passed in
validationDispatcher = iValidationDispatcher( opts, precision, executionSettings, layers );

% Verify the dispatcher has valid size respect to input and output of
% the network
if ~isempty(validationDispatcher)
    if haveDAGNetwork
        iValidateValidationDataSizeForDAGNetwork(validationDispatcher, layersGraph);
    else
        iValidateValidationDataSizeForNetwork(validationDispatcher, internalLayers);
    end
end
    
% Assert that training and validation data are consistent
iAssertTrainingAndValidationDispatcherHaveSameClasses( trainingDispatcher, validationDispatcher );

% Instantiate reporters as needed
networkInfo = nnet.internal.cnn.util.ComputeNetworkInfo(trainedNet);
[reporters, trainingPlotReporter] = iOptionalReporters(opts, internalLayers, layersMap, precision, executionSettings, networkInfo, trainingDispatcher, validationDispatcher, haveDAGNetwork);
errorState = nnet.internal.cnn.util.ErrorState();
cleanup = onCleanup(@()iFinalizePlot(trainingPlotReporter, errorState));

% Always create the info recorder (because we will reference it later) but
% only add it to the list of reporters if actually needed.
infoRecorder = iInfoRecorder( opts, internalLayers );
if nargout >= 2
    reporters.add( infoRecorder );
end

% Create a trainer to train the network with dispatcher and options
trainer = iCreateTrainer( opts, precision, reporters, executionSettings );

% Do pre-processing work required for normalizing data
trainedNet = trainer.initializeNetworkNormalizations(trainedNet, trainingDispatcher, precision, executionSettings, opts.Verbose);

% Do the training
trainedNet = trainer.train(trainedNet, trainingDispatcher);

% Do post-processing work (if any)
trainedNet = trainer.finalizeNetwork(trainedNet, trainingDispatcher);
iComputeFinalValidationResultsForPlot(trainingPlotReporter, trainedNet);
trainedNet = iPrepareNetworkForOutput(trainedNet, layersMap, haveDAGNetwork);
info = infoRecorder.Info;

% Update error state ready for the cleanup.
errorState.ErrorOccurred = false;
end

function [layers, opts, X, Y] = iParseInputArguments(varargin)
% iParseInputArguments   Parse input arguments of trainNetwork
%
% Output arguments:
%   layers  - An array of layers or a layer graph
%   opts    - An object containing training options
%   X       - Input data, this can be a data dispatcher, an image
%             datastore, a table, a numeric array or a cell array
%   Y       - Response data, this can be a numeric array or empty in case X
%             is a dispatcher, a table, an image datastore or a cell array

X = varargin{1};
if iIsADataDispatcher( X )
    % X is a custom dispatcher. The custom dispatcher api is for internal
    % use only.
    Y = [];
    layers = varargin{2};
    opts = varargin{3};
elseif iIsAnImageDatastore( X )
    iAssertOnlyThreeArgumentsForIMDS( nargin );
    Y = [];
    layers = varargin{2};
    opts = varargin{3};
elseif iIsAMiniBatchableDatastore(X)
    Y = [];
    layers = varargin{2};
    opts = varargin{3};
elseif iIsPixelLabelDatastore( X )
    Y = [];
    layers = varargin{2};
    opts = varargin{3};
elseif istable( X )
    secondArgument = varargin{2};
    if ischar(secondArgument) || iscellstr(secondArgument)
        % ResponseName syntax
        narginchk(4,4);
        responseNames = secondArgument;
        iAssertValidResponseNames(responseNames, X);
        X = iSelectResponsesFromTable( X, responseNames );
        Y = [];
        layers = varargin{3};
        opts = varargin{4};
    else
        narginchk(3,3);
        Y = [];
        layers = varargin{2};
        opts = varargin{3};
    end
elseif isnumeric( X )
    narginchk(4,4);
    Y = varargin{2};
    layers = varargin{3};
    opts = varargin{4};
elseif iscell( X )
    narginchk(4,4);
    Y = varargin{2};
    layers = varargin{3};
    opts = varargin{4};
else
    error(message('nnet_cnn:internal:cnn:util:TrainingDataErrorThrower:XIsNotValidType'));
end
end

function [X, Y] = iGetValidationDataFromOptions( opts )
X = opts.ValidationData;
if iIsADataDispatcher( X )
    % X is a custom dispatcher. The custom dispatcher api is for internal
    % use only.
    Y = [];
elseif iIsAnImageDatastore( X )
    Y = [];
elseif iIsAMiniBatchableDatastore( X )
    Y = [];
elseif istable( X )
    Y = [];
elseif iscell( X )
    Y = X{2};
    X = X{1};
else
    % Do nothing. Invalid type is already checked when creating
    % trainingOptions
end
end

function iValidateOptions( opts )
% iValidateOptions   Assert that opts is a valid training option object
if ~isa(opts, 'nnet.cnn.TrainingOptions')
    error(message('nnet_cnn:trainNetwork:InvalidTrainingOptions'))
end
end

function internalLayers = iGetInternalLayers( layers )
internalLayers = nnet.internal.cnn.layer.util.ExternalInternalConverter.getInternalLayers( layers );
end

function iValidateTrainingDataForProblem( X, Y, layers )
% iValidateTrainingDataForProblem   Assert that the input training data X
% and response Y are valid for the class of problem considered
trainingDataValidator = iTrainingDataValidator;
trainingDataValidator.validateDataForProblem( X, Y, layers );
end

function iValidateTrainingDataSizeForNetwork(dispatcher, internalLayers)
% iValidateTrainingDataSizeForNetwork   Assert that the input training data has a
% valid size for the network in use and the response size matches the
% output of the network
trainingDataValidator = iTrainingDataValidator;
trainingDataValidator.validateDataSizeForNetwork( dispatcher, internalLayers )
end

function iValidateTrainingDataSizeForDAGNetwork(dispatcher, layerGraph)
% iValidateTrainingDataSizeForDAGNetwork   Assert that the input training data has a
% valid size for the DAG network in use and the response size matches the
% output of the network
trainingDataValidator = iTrainingDataValidator;
trainingDataValidator.validateDataSizeForDAGNetwork( dispatcher, layerGraph )
end

function iValidateValidationDataForProblem( X, Y, layers )
% iValidateValidationDataForProblem   Assert that the input validation data
% X and response Y are valid for the class of problem considered
iVerifyLayersForValidation( layers );
validationDataValidator = iValidationDataValidator;
validationDataValidator.validateDataForProblem( X, Y, layers );
end

function iVerifyLayersForValidation( layers )
if iIsRNN( layers )
    error(message('nnet_cnn:trainNetwork:ValidationNotSupportedForLSTM'));
end
end

function iValidateValidationDataSizeForNetwork(dispatcher, internalLayers)
% iValidateValidationDataSizeForNetwork   Assert that the input validation
% data has a valid size for the network in use and the response size
% matches the output of the network
validationDataValidator = iValidationDataValidator;
validationDataValidator.validateDataSizeForNetwork( dispatcher, internalLayers );
end

function iValidateValidationDataSizeForDAGNetwork(dispatcher, layerGraph)
% iValidateValidationDataSizeForDAGNetwork   Assert that the input validation
% data has a valid size for the DAG network in use and the response size
% matches the output of the network
validationDataValidator = iValidationDataValidator;
validationDataValidator.validateDataSizeForDAGNetwork( dispatcher, layerGraph );
end

function iAssertTrainingAndValidationDispatcherHaveSameClasses( trainingDispatcher, validationDispatcher )
if ~isempty(validationDispatcher)
    if ~iHaveSameClassNames(trainingDispatcher, validationDispatcher)
         error(message('nnet_cnn:trainNetwork:TrainingAndValidationDifferentClasses'));
    end        
    hasDispatcherCategories = isprop(trainingDispatcher, 'Categories') && ...
        isprop(validationDispatcher, 'Categories');
    if hasDispatcherCategories        
        if ~iHaveSameOrdinality(trainingDispatcher, validationDispatcher)
            error(message('nnet_cnn:trainNetwork:TrainingAndValidationDifferentOrdinality'));        
        end
    end
end
end

function tf = iHaveSameClassNames(trainingDispatcher, validationDispatcher)
% iHaveSameClassNames   Return true if the classes in trainingDispatcher
% have the same labels as the ones in validationDispatcher. This does not
% catch the situation in which one set is a subset of the other - that
% situation will be caught when we compare the number of classes in the
% datasets to the number of classes expected by the network
tf = all(ismember(trainingDispatcher.ClassNames, validationDispatcher.ClassNames));
end

function tf = iHaveSameOrdinality(trainingDispatcher, validationDispatcher)
% iHaveSameClassNames   Return true if the Categories of the trainingDispatcher
% have the same ordinality as those of the validationDispatcher.  
tf = isequal(isordinal(trainingDispatcher.Categories), ...
    isordinal(validationDispatcher.Categories));
end

function trainingDataValidator = iTrainingDataValidator()
trainingDataValidator = nnet.internal.cnn.util.NetworkDataValidator( ...
    nnet.internal.cnn.util.TrainingDataErrorThrower );
end

function validationDataValidator = iValidationDataValidator()
validationDataValidator = nnet.internal.cnn.util.NetworkDataValidator( ...
    nnet.internal.cnn.util.ValidationDataErrorThrower );
end

function iThrowCNNException( exception )
% Wrap exception in a CNNException, which reports the error in a custom way
err = nnet.internal.cnn.util.CNNException.hBuildCustomError( exception );
throwAsCaller(err);
end

function layers = iInitializeParameters(layers, precision)
for i = 1:numel(layers)
    layers{i} = layers{i}.initializeLearnableParameters(precision);
    if isa(layers{i}, 'nnet.internal.cnn.layer.Updatable' )
        layers{i} = layers{i}.initializeDynamicParameters(precision);
    end
end
end

function externalNet = iPrepareNetworkForOutput(internalNet, layersMap, haveDAGNetwork)
% If output network is on pool, retrieve it
if isa(internalNet, 'Composite')
    spmd
        [internalNet, labWithOutput] = iPrepareNetworkForOutputOnPool(internalNet);
    end
    internalNet = internalNet{labWithOutput.Value};
else
    internalNet = iPrepareNetworkForHostPrediction(internalNet);
end

% Convert to external network for user
externalNet = iCreateExternalNetwork(internalNet, layersMap, haveDAGNetwork);
end

function [internalNet, labWithResult] = iPrepareNetworkForOutputOnPool(internalNet)
if isempty(internalNet)
    labWithResult = gop(@min, inf);
else
    labWithResult = gop(@min, labindex);
end
if labindex == labWithResult
    % Convert to host network on pool, in case client has no GPU
    internalNet = iPrepareNetworkForHostPrediction(internalNet);
end
% Only labWithResult can be returned using AutoTransfer - network is too
% big
labWithResult = distributedutil.AutoTransfer( labWithResult, labWithResult );
end

function internalNet = iPrepareNetworkForHostPrediction(internalNet)
internalNet = internalNet.prepareNetworkForPrediction();
internalNet = internalNet.setupNetworkForHostPrediction();
end

function externalNetwork = iCreateExternalNetwork(internalNetwork, layersMap, haveDAGNetwork)
% Construct an External network. We assume by this stage you have called
% internalNet.prepareNetworkForPrediction() and
% internalNet.setupNetworkForHostPrediction().
if haveDAGNetwork
    externalNetwork = DAGNetwork(internalNetwork, layersMap);
else
    % SeriesNetwork has to be constructed from the internal layers not to lose
    % information about the internal custom layers
    externalNetwork = SeriesNetwork(internalNetwork.Layers, layersMap);
    % Reset the network state, so that if the network is recurrent it is
    % configured for prediction on an arbitrary mini-batch size
    externalNetwork = externalNetwork.resetState();
end
end

function externalNetwork = iPrepareAndCreateExternalNetwork(internalNetwork, layersMap, haveDAGNetwork)
% Prepare an internal network for prediction, then create an external
% network
internalNetwork = internalNetwork.prepareNetworkForPrediction();
internalNetwork = internalNetwork.setupNetworkForHostPrediction();
externalNetwork = iCreateExternalNetwork(internalNetwork, layersMap, haveDAGNetwork);
end

function iComputeFinalValidationResultsForPlot(trainingPlotReporter, trainedNet)
trainingPlotReporter.computeFinalValidationResults(trainedNet);
end

function layersMap = iLayersMap( layers )
layersMap = nnet.internal.cnn.layer.util.InternalExternalMap( layers );
end

function infoRecorder = iInfoRecorder( opts, internalLayers )
trainingInfoContent = iTrainingInfoContent( opts, internalLayers );
infoRecorder = nnet.internal.cnn.util.traininginfo.Recorder(trainingInfoContent);
end

function aContent = iTrainingInfoContent( opts, internalLayers )
isValidationSpecified = iIsValidationSpecified(opts);

if iIsClassificationNetwork(internalLayers)
    if isValidationSpecified
        aContent = nnet.internal.cnn.util.traininginfo.ClassificationWithValidationContent;
    else
        aContent = nnet.internal.cnn.util.traininginfo.ClassificationContent;
    end
else
    if isValidationSpecified
        aContent = nnet.internal.cnn.util.traininginfo.RegressionWithValidationContent;
    else
        aContent = nnet.internal.cnn.util.traininginfo.RegressionContent;
    end
end
end

function tf = iIsClassificationNetwork(internalLayers)
tf = iIsClassificationLayer(internalLayers{end});
end

function tf = iIsClassificationLayer(internalLayer)
tf = isa(internalLayer, 'nnet.internal.cnn.layer.ClassificationLayer');
end

function layers = iMaybeStoreCategories(layers, dispatcher)
% Store categories from dispatcher if layer does not have any.
shouldSetCategories = isempty(layers{end}.Categories);
dispatcherClassNames = dispatcher.ClassNames;
% Check if the dispatcher has Categories
hasDispatcherCategories = isprop(dispatcher, 'Categories');
if shouldSetCategories 
    if hasDispatcherCategories
        labels = dispatcher.Categories;
    else   
        labels = categorical(dispatcherClassNames, dispatcherClassNames);
    end
    layers = iStoreCategories(layers, labels);
else    
    if hasDispatcherCategories && iDispatcherAndUserOrdinalityDoNotMatch(...
                dispatcher.Categories, layers{end}.Categories)
        error(message('nnet_cnn:trainNetwork:InvalidOrdinality', numel(layers)));
    end        
    userSpecifiedClassNames = layers{end}.ClassNames;
    if iDispatcherAndUserClassNamesDoNotMatch(dispatcherClassNames, userSpecifiedClassNames)
            error(message('nnet_cnn:trainNetwork:InvalidClassNames', numel(layers)));
    end
end
end

function TF = iDispatcherAndUserClassNamesDoNotMatch(...
    dispatcherClassNames, userClassNames)
% The names must match. Including the order of the name.
% If dispatcherClassNames is a row vector, transpose it before comparison.
TF = ~isequal(dispatcherClassNames(:), userClassNames);
end

function TF = iDispatcherAndUserOrdinalityDoNotMatch(...
    dispatcherCategories, userCategories)
% The two categorical arrays dispatcherCategories and userCategories
% have to have the same ordinality.
TF = ~isequal(isordinal(dispatcherCategories), isordinal(userCategories));
end

function layers = iStoreCategories(layers, labels)
layers{end}.Categories = labels;
end

function layers = iStoreResponseNames(layers, responseNames)
layers{end}.ResponseNames = responseNames;
end

function iAssertOnlyThreeArgumentsForIMDS( nArgIn )
if nArgIn~=3
    error(message('nnet_cnn:trainNetwork:InvalidNarginWithImageDatastore'));
end
end

function tf = iIsADataDispatcher(X)
tf = isa(X, 'nnet.internal.cnn.DataDispatcher');
end

function tf = iIsAMiniBatchableDatastore(X)
tf = isa(X, 'matlab.io.Datastore') && isa(X, 'matlab.io.datastore.MiniBatchable');
end

function tf = iIsParallelExecutionEnvironment(executionEnvironment)
tf = ismember( executionEnvironment, {'multi-gpu', 'parallel'} );
end

function executionSettings = iSetupExecutionEnvironment( opts, isRNN, X )
% Detect CPU/GPU/multiGPU/parallel training, and set up environment
% appropriately
backgroundPrefetch = iUseBackgroundPrefetch(X);
executionSettings = struct( ...
    'executionEnvironment', 'cpu', ...
    'useParallel', false, ...
    'backgroundPrefetch', backgroundPrefetch, ...
    'workerLoad', 1 );
isParallel = iIsParallelExecutionEnvironment( opts.ExecutionEnvironment );
if ( isParallel && isRNN )
    error(message('nnet_cnn:trainNetwork:InvalidRNNExecutionEnvironment'));
end
if isParallel
    [executionSettings.useParallel, executionSettings.workerLoad, executionSettings.backgroundPrefetch] = ...
        iSetupAndValidateParallel( opts.ExecutionEnvironment, opts.WorkerLoad, backgroundPrefetch);
end

GPUShouldBeUsed = nnet.internal.cnn.util.GPUShouldBeUsed( ...
    opts.ExecutionEnvironment, executionSettings.workerLoad );
if GPUShouldBeUsed
    executionSettings.executionEnvironment = 'gpu';
end
end

function [useParallel, workerLoad, backgroundPrefetch] = iSetupAndValidateParallel( executionEnvironment, workerLoad, backgroundPrefetch )
% Pool and work-per-worker setup and validation
nnet.internal.cnn.util.validatePCTIsInstalled(executionEnvironment);
[useParallel, isMultiGpu, pool] = iValidateParallelPool( executionEnvironment, backgroundPrefetch );
if useParallel
    [workerLoad, backgroundPrefetch] = iValidateWorkerLoad( isMultiGpu, pool, workerLoad, backgroundPrefetch );
end
end

function [useParallel, isMultiGpu, pool] = iValidateParallelPool( executionEnvironment, backgroundPrefetch )
% Detect parallel training, open a pool if necessary, and validate that
% pool
useParallel = true;
pool = gcp('nocreate');

% Multi-GPU (local parallel pool)
% Expect a local pool to be open, or open one with one worker per GPU
isMultiGpu = false;
if string(executionEnvironment) == "multi-gpu"
    isMultiGpu = true;
    
    % Check that there are supported GPUs
    numGpus = gpuDeviceCount();
    if numGpus == 0
        error(message('parallel:gpu:device:NoCUDADevice'));
    end
    if ~isempty(pool)
        % Check that the open pool is local
        if ~isa( pool.Cluster, 'parallel.cluster.Local' )
            error(message('nnet_cnn:trainNetwork:ExpectedLocalPool'));
        end
    else
        % If no pool is open and there is only one supported GPU, we
        % should train as normal, on the client, without opening a
        % pool. User can force training to happen on a pool by opening
        % it themselves.
        if numGpus == 1
            isMultiGpu = false;
            useParallel = false;
            return;
        else
            % Check that the default cluster profile is local
            defaultProfileName = parallel.defaultClusterProfile();
            defaultProfileType = parallel.internal.settings.ProfileExpander.getClusterType( defaultProfileName );
            if defaultProfileType == parallel.internal.types.SchedulerType.Local
                % Open the default cluster with numGpus workers, or the
                % default number of workers if using background prefetch
                % Account for the possibility that user has changed the
                % default local profile to have fewer workers
                clust = parcluster( defaultProfileName );
                numWorkers = numGpus;
                if backgroundPrefetch || clust.NumWorkers < numGpus
                    numWorkers = clust.NumWorkers;
                end
                % Open pool. We need SPMD enabled and doing it when opening
                % the pool leads to faster communication.
                pool = parpool( clust, numWorkers, 'SpmdEnabled', true );
            else
                error(message('nnet_cnn:trainNetwork:MultiGpuRequiresDefaultLocal', defaultProfileName));
            end
        end
    end
    
    % General parallel pool
    % Expect a pool to be open, or open the default pool
else
    if isempty(pool)
        % Error if user has disabled auto-pool creation
        s = settings;
        if s.parallel.client.pool.AutoCreate.ActiveValue == 0
            error(message('nnet_cnn:trainNetwork:ParallelAutoOpenDisabled'));
        end
        % Open pool using default profile
        pool = parpool( 'SpmdEnabled', true );
    end
end

% Check that SPMD is enabled in the current pool
if ~pool.SpmdEnabled
    error(message('nnet_cnn:trainNetwork:SPMDDisabled'));
end
end

function [workerLoad, backgroundPrefetch] = iValidateWorkerLoad( isMultiGpu, pool, userWorkerLoad, backgroundPrefetch )
% Given a parallel pool, modify the workerLoad settings to disable any
% workers that cannot access GPUs - unless there are no GPUs, in which case
% assume training on all pool CPUs

% Initialize workerLoad, using input user settings if provided
numWorkers = pool.NumWorkers;
useDefaultWorkerLoad = false;
if ~isempty(userWorkerLoad)
    % Validate user input
    if ~isscalar(userWorkerLoad) && length(userWorkerLoad) ~= numWorkers
        error(message('nnet_cnn:trainNetwork:InvalidWorkerLoad'));
    end
else
    userWorkerLoad = ones( 1, numWorkers );
    useDefaultWorkerLoad = true;
end

% Collect machine and device info about each worker
spmd
    [hostname, deviceIndex] = iGetHostnameAndDeviceIndex();
end
deviceIndex = [ deviceIndex{:} ];
[~, order, hostIds] = unique({ hostname{:} }, 'sorted');
numNodes = numel( order );
numWorkersPerNode = accumarray(hostIds, 1);

% Deal with CPU-only clusters as if they all have unique GPUs, so no
% workers are eliminated
cpuCluster = all( deviceIndex == 0 );
if cpuCluster
    deviceIndex = 1:numWorkers;
end
% Default worker load for CPU-only clusters with background prefetch is
% for all but one on each node to be used for training computation
cpuBackgroundDefault = false;
if cpuCluster && useDefaultWorkerLoad && backgroundPrefetch
    cpuBackgroundDefault = true;
end

% For scalar worker load, determine the loads based on the environment on
% each host
if isscalar(userWorkerLoad) || cpuBackgroundDefault
    % Map from values grouped by host back to workers
    [~, I] = sort( hostIds );
    originalIndices = 1:numWorkers;
    originalIndices = originalIndices(I);
    
    % Calculate how many workers on each host should be doing training
    % computation
    if cpuBackgroundDefault
        computeWorkersPerNode = max( 1, numWorkersPerNode - 1 );
    elseif userWorkerLoad < 1
        computeWorkersPerNode = ceil( numWorkersPerNode * userWorkerLoad );
    else
        computeWorkersPerNode = min( numWorkersPerNode, floor( userWorkerLoad ) );
    end
    
    % Start with a load of 1 on every worker, then zero the ones that are
    % not compute workers. This can be vectorized but the code is complex
    % and slow.
    userWorkerLoad = ones( 1, numWorkers );
    assert( numNodes == numel( numWorkersPerNode ) );
    i = 1;
    for n = 1:numNodes
        userWorkerLoad((i+computeWorkersPerNode(n)):(i+numWorkersPerNode(n)-1)) = 0;
        i = i + numWorkersPerNode(n);
    end
    
    % Scatter the resulting loads out to the correct locations in the pool.
    % Typically this will do nothing because the hosts are grouped and
    % ordered.
    userWorkerLoad = userWorkerLoad(originalIndices);
end

% Create the final worker load by copying the user settings, then stripping
% out non-unique device/host combinations in the active workers
workerLoad = userWorkerLoad;
hostnameDevice = [ hostIds(:) deviceIndex(:) ];
% Eliminate workers that are disabled manually or have no GPU. CPU-only
% clusters will be treated as if all workers have a unique GPU.
maskDisabled = workerLoad == 0 | deviceIndex == 0;
hostnameDevice(maskDisabled,1) = 0;
hostnameDevice(maskDisabled,2) = 0;
% Now reduce to the remaining unique combinations
[~, uniqueIndices] = unique(hostnameDevice, 'rows', 'stable');
% Mask superfluous and no-GPU workers
mask = true( 1, numWorkers );
mask(uniqueIndices) = false;
mask(maskDisabled) = true;
% Apply mask to existing settings
workerLoad(mask) = 0;

% Special case: the result of the above is that all workers are idle, which
% can only happen in a hybrid pool where all the workers on GPU hosts have
% been explicitly disabled and only workers on CPU hosts are left. In this
% case training will happen on the CPUs of the remaining workers.
if all( workerLoad == 0 )
    workerLoad = userWorkerLoad;
    return;
end

% Save the warning state before emitting WorkerLoad warnings
warnState = warning('query', 'backtrace');
cleanupObj = onCleanup(@()warning(warnState));
warning off backtrace;

% Report a warning if there are idle workers as a consequence of GPU
% sharing
disabledLabsMask = (userWorkerLoad > 0) & (workerLoad == 0);
numNewIdleWorkers = sum( disabledLabsMask );
if numNewIdleWorkers > 0
    problemLabsStr = mat2str( find( disabledLabsMask ) );
    if ~backgroundPrefetch
        if isMultiGpu
            warning(message('nnet_cnn:trainNetwork:SomeWorkersIdleLocal', problemLabsStr));
        else
            warning(message('nnet_cnn:trainNetwork:SomeWorkersIdleCluster', problemLabsStr));
        end
    elseif ~useDefaultWorkerLoad
        if isMultiGpu
            warning(message('nnet_cnn:trainNetwork:SomeComputeWorkersDisabledLocal', problemLabsStr));
        else
            warning(message('nnet_cnn:trainNetwork:SomeComputeWorkersDisabledCluster', problemLabsStr));
        end
    end
end

% Validate background prefetch. If this is being used, there must be at
% least one prefetch worker, otherwise it will be disabled.
if backgroundPrefetch
    computeWorkers = double( workerLoad > 0 );
    computeWorkersPerNode = accumarray( hostIds, computeWorkers );
    nodesWithNoBackgroundWorkers = computeWorkersPerNode == numWorkersPerNode;
    if any(nodesWithNoBackgroundWorkers)
        if isMultiGpu
            warning(message('nnet_cnn:trainNetwork:BackgroundPrefetchDisabledLocal'));
            backgroundPrefetch = false;
        else
            whichNodes = find(nodesWithNoBackgroundWorkers);
            allLabs = 1:numWorkers;
            labsOnFirstProblemNode = allLabs(hostIds == whichNodes(1));
            workerLoadAfterDisablingNodes = workerLoad;
            workerLoadAfterDisablingNodes(hostIds == whichNodes) = 0;
            if all(workerLoadAfterDisablingNodes == 0)
                % Disable background because there are no background
                % workers on any hosts which have compute workers
                warning( message( 'nnet_cnn:trainNetwork:BackgroundPrefetchDisabledCluster' ) );
                backgroundPrefetch = false;
            else
                % Warn that hosts with no background workers will be
                % disabled
                warning( message( 'nnet_cnn:trainNetwork:SomeNodesDisabledCluster', ...
                    mat2str(labsOnFirstProblemNode) ) );
                % Set workerLoad to zero on all these labs.
                workerLoad = workerLoadAfterDisablingNodes;
            end
        end
    end
end

end

function dispatcher = iCreateTrainingDataDispatcher(X, Y, opts, precision, executionSettings, layers)
% Create a dispatcher.
dispatcher = nnet.internal.cnn.DataDispatcherFactory.createDataDispatcher( ...
    X, Y, opts.MiniBatchSize, 'discardLast', precision, executionSettings,...
    opts.Shuffle, opts.SequenceLength, opts.SequencePaddingValue, layers);
end

function dispatcher = iCreateValidationDataDispatcher(X, Y, opts, precision, trainingExecutionSettings)
% iCreateValidationDataDispatcher   Create a dispatcher for validation data

% Validation execution settings
executionSettings = iSetValidationExecutionSettings(trainingExecutionSettings);
dispatcher = nnet.internal.cnn.DataDispatcherFactory.createDataDispatcher( ...
    X, Y, opts.MiniBatchSize, 'truncateLast', precision, executionSettings, opts.Shuffle, 'longest', 0);
end

function executionSettings = iSetValidationExecutionSettings(trainingExecutionSettings)
% Copy training settings for use with validation
executionSettings = trainingExecutionSettings;
% If the training execution environment is parallel, prefetching cannot be
% used by the validation dispatcher
if trainingExecutionSettings.useParallel
    executionSettings.backgroundPrefetch = false;
end
% Validation dispatcher cannot be parallel
executionSettings.useParallel = false;
end

function [reporter, trainingPlotReporter] = iOptionalReporters(opts, internalLayers, layersMap, precision, executionSettings, networkInfo, trainingDispatcher, validationDispatcher, haveDAGNetwork)
% iOptionalReporters   Create a vector of Reporters based on the given
% training options and the network type
%
% See also: nnet.internal.cnn.util.VectorReporter
reporter = nnet.internal.cnn.util.VectorReporter();

isValidationSpecified = iIsValidationSpecified(opts);

isAClassificationNetwork = iIsClassificationNetwork(internalLayers);
if opts.Verbose
    % If verbose is true, add a progress displayer
    if isAClassificationNetwork
        if isValidationSpecified
            columnStrategy = nnet.internal.cnn.util.ClassificationValidationColumns;
        else
            columnStrategy = nnet.internal.cnn.util.ClassificationColumns;
        end
    else
        if isValidationSpecified
            columnStrategy = nnet.internal.cnn.util.RegressionValidationColumns;
        else
            columnStrategy = nnet.internal.cnn.util.RegressionColumns;
        end
    end
    progressDisplayerFrequency = opts.VerboseFrequency;
    if isValidationSpecified
        progressDisplayerFrequency = [progressDisplayerFrequency opts.ValidationFrequency];
    end
    progressDisplayer = nnet.internal.cnn.util.ProgressDisplayer(columnStrategy);
    progressDisplayer.Frequency = progressDisplayerFrequency;
    reporter.add( progressDisplayer );
end

if isValidationSpecified
    % Create a validation reporter
    validationReporter = iValidationReporter( validationDispatcher, precision, executionSettings, opts.ValidationFrequency, opts.ValidationPatience, opts.Shuffle );
    reporter.add( validationReporter );
end

if ~isempty( opts.CheckpointPath )
    checkpointSaver = nnet.internal.cnn.util.CheckpointSaver( opts.CheckpointPath );
    checkpointSaver.ConvertorFcn = @(net)iPrepareAndCreateExternalNetwork(net, layersMap, haveDAGNetwork);
    reporter.add( checkpointSaver );
end

if ~isempty( opts.OutputFcn )
    userCallbackReporter = nnet.internal.cnn.util.UserCallbackReporter( opts.OutputFcn );
    reporter.add( userCallbackReporter );
end

if strcmp( opts.Plots, 'training-progress' )
    if isdeployed
        error(message('nnet_cnn:internal:cnn:ui:trainingplot:TrainingPlotNotDeployable'))
    end
    if ~isValidationSpecified
        validationReporter = nnet.internal.cnn.util.EmptyValidationReporter();   % To be used only by the trainingPlotReporter
    end
    trainingPlotReporter = iCreateTrainingPlotReporter(isAClassificationNetwork, executionSettings, opts, internalLayers, networkInfo, trainingDispatcher, isValidationSpecified, validationReporter);
    reporter.add( trainingPlotReporter );
else
    trainingPlotReporter = nnet.internal.cnn.util.EmptyPlotReporter();
end
end

function trainingPlotReporter = iCreateTrainingPlotReporter(isAClassificationNetwork, executionSettings, opts, internalLayers, networkInfo, trainingDispatcher, isValidationSpecified, validationReporter)
hasVariableNumItersPerEpoch = iHasVariableNumItersEachEpoch(opts, internalLayers);
if hasVariableNumItersPerEpoch
    epochDisplayer = nnet.internal.cnn.ui.axes.EpochDisplayHider();
    determinateProgress = nnet.internal.cnn.ui.progress.DeterminateProgressText();
    tableDataFactory = nnet.internal.cnn.ui.info.VariableEpochSizeTextTableDataFactory();
else
    epochDisplayer = nnet.internal.cnn.ui.axes.EpochAxesDisplayer();
    determinateProgress = nnet.internal.cnn.ui.progress.DeterminateProgressBar();
    tableDataFactory = nnet.internal.cnn.ui.info.TextTableDataFactory();
end

% create the view
legendLayout = nnet.internal.cnn.ui.layout.Legend();
textLayout = nnet.internal.cnn.ui.layout.TextTable();
trainingPlotView = nnet.internal.cnn.ui.TrainingPlotViewHG(determinateProgress, legendLayout, textLayout);

% create the presenter
if isAClassificationNetwork
    axesFactory = nnet.internal.cnn.ui.factory.ClassificationAxesFactory();
    metricRowDataFactory = nnet.internal.cnn.ui.info.ClassificationMetricRowDataFactory();
else
    axesFactory = nnet.internal.cnn.ui.factory.RegressionAxesFactory();
    metricRowDataFactory = nnet.internal.cnn.ui.info.RegressionMetricRowDataFactory();
end
executionInfo = nnet.internal.cnn.ui.ExecutionInfo(executionSettings.executionEnvironment, executionSettings.useParallel, opts.LearnRateScheduleSettings.Method, opts.InitialLearnRate);
validationInfo = nnet.internal.cnn.ui.ValidationInfo(isValidationSpecified, opts.ValidationFrequency, opts.ValidationPatience);
%
watch = nnet.internal.cnn.ui.adapter.Stopwatch();
stopReasonRowDataFactory = nnet.internal.cnn.ui.info.StopReasonRowDataFactory();
preprocessingDisplayer = iCreatePreprocessingDisplayer(networkInfo);
helpLauncher = nnet.internal.cnn.ui.info.TrainingPlotHelpLauncher();
epochInfo = iCreateEpochInfo(opts, trainingDispatcher);
dialogFactory = nnet.internal.cnn.ui.DialogFactory();
trainingPlotPresenter = nnet.internal.cnn.ui.TrainingPlotPresenterWithDialog(...
    trainingPlotView, tableDataFactory, metricRowDataFactory, stopReasonRowDataFactory, preprocessingDisplayer, dialogFactory, ...
    axesFactory, epochDisplayer, helpLauncher, watch, executionInfo, validationInfo, epochInfo);

% create the reporter
summaryFactory = nnet.internal.cnn.util.SummaryFactory();
trainingPlotReporter = nnet.internal.cnn.util.TrainingPlotReporter(trainingPlotPresenter, validationReporter, summaryFactory, epochInfo);
end

function iFinalizePlot(trainingPlotReporter, errorState)
trainingPlotReporter.finalizePlot(errorState.ErrorOccurred);
end

function epochInfo = iCreateEpochInfo(opts, trainingDispatcher)
epochInfo = nnet.internal.cnn.ui.EpochInfo(opts.MaxEpochs, trainingDispatcher.NumObservations, opts.MiniBatchSize);
end

function preprocessingDisplayer = iCreatePreprocessingDisplayer(networkInfo)
if networkInfo.ShouldImageNormalizationBeComputed
    dialogFactory = nnet.internal.cnn.ui.DialogFactory();
    preprocessingDisplayer = nnet.internal.cnn.ui.PreprocessingDisplayerDialog(dialogFactory);
else
    preprocessingDisplayer = nnet.internal.cnn.ui.PreprocessingDisplayerEmpty();
end
end

function tf = iHasVariableNumItersEachEpoch(opts, internalLayers)
isRNN = isa(internalLayers{1}, 'nnet.internal.cnn.layer.SequenceInput');
hasCustomSequenceLength = isnumeric(opts.SequenceLength);
tf = isRNN && hasCustomSequenceLength;
end

function validationDispatcher = iValidationDispatcher(opts, precision, executionSettings, layers)
% iValidationDispatcher   Get validation data and create a dispatcher for it. Validate the
% data for the current problem and w.r.t. the current architecture.

% Return empty if no validation data was specified
if ~iIsValidationSpecified(opts)
    validationDispatcher = [];
else
    % There is no need to convert datastore into table, since validation
    % will be computed only on one worker
    [XVal, YVal] = iGetValidationDataFromOptions( opts );
    iValidateValidationDataForProblem( XVal, YVal, layers );
    % Create a validation dispatcher
    validationDispatcher = iCreateValidationDataDispatcher(XVal, YVal, opts, precision, executionSettings);
end
end

function tf = iIsValidationSpecified(opts)
tf = ~isempty(opts.ValidationData);
end

function validator = iValidationReporter(validationDispatcher, precision, executionEnvironment, frequency, patience, shuffle)
validator = nnet.internal.cnn.util.ValidationReporter(validationDispatcher, precision, executionEnvironment, frequency, patience, shuffle);
end

function trainer = iCreateTrainer(opts, precision, reporters, executionSettings)
if ~executionSettings.useParallel
    trainer = nnet.internal.cnn.Trainer(opts, precision, reporters, executionSettings);
else
    trainer = nnet.internal.cnn.ParallelTrainer(opts, precision, reporters, executionSettings);
end
end

function tf = iIsAnImageDatastore(x)
tf = isa(x, 'matlab.io.datastore.ImageDatastore');
end

function iAssertValidResponseNames(responseNames, tbl)
% iAssertValidResponseNames   Assert that the response names are variables
% of the table and they do not refer to the first column.
variableNames = tbl.Properties.VariableNames;
refersToFirstColumn = ismember( variableNames(1), responseNames );
responseNamesAreAllVariables = all( ismember(responseNames,variableNames) );
if refersToFirstColumn || ~responseNamesAreAllVariables
    error(message('nnet_cnn:trainNetwork:InvalidResponseNames'))
end
end

function resTbl = iSelectResponsesFromTable(tbl, responseNames)
% iSelectResponsesFromTable   Return a new table with only the first column
% (predictors) and the variables specified in responseNames.
variableNames = tbl.Properties.VariableNames;
varTF = ismember(variableNames, responseNames);
% Make sure to select predictors (first column) as well
varTF(1) = 1;
resTbl = tbl(:,varTF);
end

function [hostid, deviceIndex] = iGetHostnameAndDeviceIndex()
hostid = parallel.internal.general.HostNameUtils.getLocalHostAddress();
try
    if nnet.internal.cnn.util.isGPUCompatible()
        deviceIndex = parallel.internal.gpu.currentDeviceIndex();
    else
        deviceIndex = 0;
    end
catch
    deviceIndex = 0;
end
end

function tf = iIsPixelLabelDatastore(x)
tf = isa(x, 'matlab.io.datastore.PixelLabelDatastore');
end

function internalNetwork = iCreateInternalNetwork( lgraph, internalLayers, haveDAGNetwork )
if haveDAGNetwork
    internalLayerGraph = iExternalToInternalLayerGraph( lgraph );
    internalLayerGraph.Layers = internalLayers;
    topologicalOrder = extractTopologicalOrder( lgraph );
    internalNetwork = nnet.internal.cnn.DAGNetwork(internalLayerGraph, topologicalOrder);
else
    internalNetwork = nnet.internal.cnn.SeriesNetwork(internalLayers);
end
end

function internalLayerGraph = iExternalToInternalLayerGraph( externalLayerGraph )
internalLayers = iGetInternalLayers( externalLayerGraph.Layers );
hiddenConnections = externalLayerGraph.HiddenConnections;
internalConnections = iHiddenToInternalConnections( hiddenConnections );
internalLayerGraph = nnet.internal.cnn.LayerGraph(internalLayers, internalConnections);
end

function internalConnections = iHiddenToInternalConnections( hiddenConnections )
internalConnections = nnet.internal.cnn.util.hiddenToInternalConnections( hiddenConnections );
end

function haveDAGNetwork = iHaveDAGNetwork(lgraph)
haveDAGNetwork = isa(lgraph,'nnet.cnn.LayerGraph');
end

function tf = iIsRNN( layers )
internalLayers = nnet.internal.cnn.layer.util.ExternalInternalConverter.getInternalLayers( layers );
hasSequenceInput = nnet.internal.cnn.util.isRNN( internalLayers );
hasRNNLayers = any( cellfun(@(l)isa(l, 'nnet.internal.cnn.layer.Updatable' ), internalLayers ) );
tf = hasSequenceInput || hasRNNLayers;
end

function analysis = iInferParameters(layersOrGraph)
[~, analysis] = nnet.internal.cnn.layer.util.inferParameters(layersOrGraph);
end

function tf = iUseBackgroundPrefetch(X)
tf = (isa(X, 'matlab.io.datastore.BackgroundDispatchable') && X.DispatchInBackground) || ...
    (isa(X, 'nnet.internal.cnn.BackgroundCapableDispatcher') && X.RunInBackground);
end