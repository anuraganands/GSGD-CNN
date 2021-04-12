classdef SeriesNetwork
    % SeriesNetwork   A neural network with layers arranged in a series
    %
    %   A series network is one where the layers are arranged one after the
    %   other. There is a single input and a single output.
    %
    %   SeriesNetwork properties:
    %       Layers                  - The layers of the network.
    %
    %   SeriesNetwork methods:
    %       predict                 - Run the network on input data.
    %       classify                - Classify data with a network.
    %       activations             - Compute specific network layer activations.
    %       predictAndUpdateState   - Predict on data and update network state.
    %       classifyAndUpdateState  - Classify data and update network state.
    %       resetState              - Reset network state.
    %
    %   Example:
    %       Train a convolutional neural network on some synthetic images
    %       of handwritten digits. Then run the trained network on a test
    %       set, and calculate the accuracy.
    %
    %       [XTrain, TTrain] = digitTrain4DArrayData;
    %
    %       layers = [ ...
    %           imageInputLayer([28 28 1])
    %           convolution2dLayer(5,20)
    %           reluLayer()
    %           maxPooling2dLayer(2,'Stride',2)
    %           fullyConnectedLayer(10)
    %           softmaxLayer()
    %           classificationLayer()];
    %       options = trainingOptions('sgdm', 'Plots', 'training-progress');
    %       net = trainNetwork(XTrain, TTrain, layers, options);
    %
    %       [XTest, TTest] = digitTest4DArrayData;
    %
    %       YTest = classify(net, XTest);
    %       accuracy = sum(YTest == TTest)/numel(TTest)
    %
    %   See also trainNetwork.
    
    %   Copyright 2015-2017 The MathWorks, Inc.
    
    properties(Dependent, SetAccess = private)
        % Layers   The layers of the network
        %   The array of layers for the network. Each layer has different
        %   properties depending on what type of layer it is. The first
        %   layer is always an input layer, and the last layer is always an
        %   output layer.
        Layers
    end
    
    properties(Access = private)
        % PrivateNetwork   The private network object containing the
        %                  private layers
        PrivateNetwork
        
        % LayersMap   Object to hold mapping between internal and external
        %             layers
        %             (nnet.internal.cnn.layer.util.InternalExternalMap)
        LayersMap
        
        % IsClassificationNetwork Flag used to assert if the network is a
        %                         classification network or not
        IsClassificationNetwork;
        
        % Categories Categories of the network (those categories should not
        %            change during training/prediction)
        Categories;
        
        % PredictEnvironment Handle cache to store prediction environment
        %                    variables. The internal value is a struct
        %                    composed by the following fields:
        %                    - ExecutionEnvironment
        %                    - GPUShouldBeUsed
        %                    - PredictNetwork
        PredictEnvironment;
        
        % IsRNN   Flag which is true if the network is a recurrent neural
        %         network and false otherwise
        IsRNN
        
        % StatefulLayers   Logical vector of length L, where L is the
        %                  number of layers in the network. Entries are
        %                  true if the layer is Stateful. Elsewhere the
        %                  entries are false.
        StatefulLayers
        
        % ReturnSequence   Flag which is true if the network is a recurrent
        %                  neural network with sequence output, and false
        %                  otherwise
        ReturnSequence
        
        OutputSizeCached;
    end
    
    properties(Access = private, Dependent)
        % InputSize    Size of the network input as stored in the input
        % layer
        InputSize
        
        % Outputsize    Size of the network output obtained by forward
        % propagating the input size through the network
        OutputSize
    end
    
    methods
        function val = get.InputSize(this)
            val = this.PrivateNetwork.Layers{1}.InputSize;
        end
        
        function val = get.OutputSize(this)
            if this.OutputSizeCached.isEmpty
                internalLayers = this.PrivateNetwork.Layers;
                outputLayerIdx = numel( internalLayers );
                val = iDetermineLayerOutputSize( internalLayers, outputLayerIdx );
                this.OutputSizeCached.fillCache(val);
            else
                val = this.OutputSizeCached.Value;
            end
        end
        
        function layers = get.Layers(this)
            layers = this.LayersMap.externalLayers( this.PrivateNetwork.Layers );
        end
    end
    
    methods(Access = public, Hidden)
        function this = SeriesNetwork(varargin)
            % SeriesNetwork    Constructor for series network
            
            % Overloaded constructor for different function signatures
            switch nargin
                case 1
                    this = SeriesNetworkFromLayers( this, varargin{:} );
                case 2
                    this = SeriesNetworkFromLayersAndMap( this, varargin{:} );
                otherwise
                    narginchk(1,2)
            end
        end
    end
    
    methods(Access = private, Hidden)
        function this = SeriesNetworkFromLayers(this, layers)
            % SeriesNetworkFromLayers    Constructor for series network
            %
            % layers is an heterogeneous array of nnet.cnn.layer.Layer
            
            % Create an internal to external layers map
            layersMap = iLayersMap( layers );
            
            % Retrieve the internal layers
            internalLayers = nnet.cnn.layer.Layer.getInternalLayers( layers );
            
            % Create the network
            this = SeriesNetworkFromLayersAndMap(this, internalLayers, layersMap);
        end
        
        function this = SeriesNetworkFromLayersAndMap(this, internalLayers, layersMap)
            % SeriesNetworkFromLayersAndMap    Constructor for series
            % network
            %
            % internalLayers is a cell array of nnet.internal.cnn.layer.Layer
            % layersMap is a nnet.internal.cnn.layer.util.InternalExternalMap
            
            % Create an internal to external layers map
            this.LayersMap = layersMap;
            
            % Create the network
            this.PrivateNetwork = nnet.internal.cnn.SeriesNetwork( internalLayers );
            
            % Determine if it is a classification network
            this.IsClassificationNetwork = iIsAClassificationNetwork( internalLayers );
            
            % The predict environment is empty at the beginning
            this.PredictEnvironment = nnet.internal.cnn.layer.learnable.CacheHandle();
                        
            % Set the appropriate categories (thus enforcing the original
            % order and ordinality)
            this.Categories = iCategories(this.PrivateNetwork);
            
            % Determine if the network is recurrent
            this.IsRNN = nnet.internal.cnn.util.isRNN( internalLayers );
            
            % Determine if the network has returns a sequence and the
            % layers which have state
            [this.ReturnSequence, this.StatefulLayers] = nnet.internal.cnn.util.returnsSequence( internalLayers, this.IsRNN );
            
            % Make sure that this.Categories is a column vector
            if isrow(this.Categories)
                this.Categories = this.Categories';
            end
            
            this.OutputSizeCached = nnet.internal.cnn.layer.learnable.CacheHandle();
        end
    end
    
    methods(Access = public)
        function Y = predict(this, data, varargin)
            % predict   Make predictions on data with a network
            %
            %   Y = predict(net, X) will compute predictions of the network
            %   net on the data X. The format of X will depend on the input
            %   layer for the network.
            %
            %   For an image input layer, X may be:
            %       - A single image.
            %       - A 4D array of images, where the first three
            %         dimensions index the height, width and channels of an
            %         image, and the fourth dimension indexes the
            %         individual images.
            %       - An image datastore.
            %       - A matlab.io.datastore.MiniBatchDatastore.
            %       - A table, where the first column contains either image
            %         paths or images.
            %
            %   For a network with a sequence input layer, X may be:
            %       - A single time series represented by a numeric array
            %       of size D-by-S, where D is the number of data points
            %       per timestep, and S is the total number of timesteps.
            %       - A cell array of multiple time series, of size N-by-1
            %       where N is the number of observations, and each element
            %       of the cell array contains a D-by-S time series.
            %
            %   For a classification problem, Y will contain the predicted
            %   scores, arranged in an N-by-K matrix, where N is the number
            %   of observations, and K is the number of classes. If the
            %   output of the network is a sequence, scores will be an
            %   N-by-1 cell array and each element of scores will be a
            %   K-by-S numeric array, where S is the sequence length of the
            %   output. If the number of observations in X is one, scores
            %   will be a K-by-S numeric array.
            %
            %   For a regression problem, Y will contain the predicted
            %   responses, arranged in an N-by-R matrix, where N is the
            %   number of observations, and R is the number of responses,
            %   or in an H-by-W-by-C-by-N 4D array, where N is the number
            %   of observations and H-by-W-by-C is the size of a single
            %   response.
            %
            %   Y = predict(net, X, 'PARAM1', VAL1, ...) will compute
            %   predictions with the following optional name/value pairs:
            %
            %       'MiniBatchSize'     - The size of the mini-batches for
            %                             computing predictions. Larger
            %                             mini-batch sizes lead to faster
            %                             predictions, at the cost of more
            %                             memory. The default is 128.
            %
            %       'ExecutionEnvironment'
            %                           - The execution environment for the
            %                             network. This determines what
            %                             hardware resources will be used
            %                             to run the network.
            %                               - 'auto' - Use a GPU if it is
            %                                 available, otherwise use the
            %                                 CPU.
            %                               - 'gpu' - Use the GPU. To use a
            %                                 GPU, you must have Parallel
            %                                 Computing Toolbox(TM), and a
            %                                 CUDA-enabled NVIDIA GPU with
            %                                 compute capability 3.0 or
            %                                 higher. If a suitable GPU is
            %                                 not available, predict
            %                                 returns an error message.
            %                               - 'cpu' - Use the CPU.
            %                             The default is 'auto'.
            %
            %       'SequenceLength'
            %                           - Strategy to determine the length
            %                           of the sequences used per
            %                           mini-batch. Options are:
            %                               - 'longest' to pad all
            %                               sequences in a batch to the
            %                               length of the longest sequence.
            %                               - 'shortest' to truncate all
            %                               sequences in a batch to the
            %                               length of the shortest
            %                               sequence. 
            %                               -  Positive integer - Pad
            %                               sequences to the have same
            %                               length as the longest sequence,
            %                               then split into smaller
            %                               sequences of the specified
            %                               length. If splitting occurs,
            %                               then the function creates extra
            %                               mini-batches.
            %                           The default is 'longest'.
            %
            %       'SequencePaddingValue'
            %                           - Scalar value used to pad
            %                           sequences where necessary. The
            %                           default is 0.
            %
            %   See also SeriesNetwork/classify, SeriesNetwork/activations.
            
            % Set desired precision
            precision = nnet.internal.cnn.util.Precision('single');
            
            [miniBatchSize, executionEnvironment, sequenceLength, sequencePaddingValue] = iParseAndValidatePredictInputs( varargin{:} );
            if this.PredictEnvironment.isEmpty || ...
                    ~strcmp(this.PredictEnvironment.Value.ExecutionEnvironment, executionEnvironment)
                % If the cache is empty or the user is using another execution
                % environment, then fill the cache
                predictEnvironment = this.getPredictionEnvironment(executionEnvironment);
                this.PredictEnvironment.fillCache(predictEnvironment);
            end
            
            % Get the environment from the cache
            predictEnvironment = this.PredictEnvironment.Value;
            predictNetwork = predictEnvironment.PredictNetwork;
            GPUShouldBeUsed = predictEnvironment.GPUShouldBeUsed;
            
            if ~this.IsRNN
                singleNumericObservation = isnumeric(data) && (ndims(data) < 4);
                if singleNumericObservation
                    iAssertInputDataIsValidForPredict(data, this.InputSize, this.IsRNN);
                    data = precision.cast(data);
                    Y = iPredictSingleNumericObserv(predictNetwork,data,GPUShouldBeUsed);
                else
                    dispatcher = iDataDispatcher(data, miniBatchSize, precision, ...
                        sequenceLength, sequencePaddingValue, this.Layers);
                    iAssertInputDataIsValidForPredict(dispatcher, this.InputSize, this.IsRNN);
                    
                    Y = precision.zeros([this.OutputSize dispatcher.NumObservations]);
                    
                    % Use the dispatcher to run the network on the data
                    dispatcher.start();
                    while ~dispatcher.IsDone
                        [X, ~, i] = dispatcher.next();
                        Y(:,:,:,i) = iPredictSingleNumericObserv(predictNetwork,X,GPUShouldBeUsed);
                    end
                end
                
                Y = iFormatPredictions(Y);
                
            else
                Y = this.predictRNN(data, miniBatchSize, precision, sequenceLength, ...
                    sequencePaddingValue, predictNetwork, GPUShouldBeUsed);
            end
        end
        
        function [this, Y] = predictAndUpdateState(this, data, varargin)
            % predictAndUpdateState   Make predictions on data with a
            % network and update the network state
            %
            %   [net, Y] = predictAndUpdateState(net, X) computes
            %   predictions of the network net on the data X, and updates
            %   the state of the network net.
            %
            %   This method only supports networks with recurrent layers.
            %
            %   The format of X may be:
            %       - A single time series represented by a numeric array
            %       of size D-by-S, where D is the number of data points
            %       per timestep, and S is the total number of timesteps.
            %       - A cell array of multiple time series, of size N-by-1
            %       where N is the number of observations, and each element
            %       of the cell array contains a D-by-S time series.
            %
            %   For a classification problem, Y will contain the predicted
            %   scores, arranged in an N-by-K matrix, where N is the number
            %   of observations, and K is the number of classes. If the
            %   output of the network is a sequence, scores will be an
            %   N-by-1 cell array and each element of scores will be a
            %   K-by-S numeric array, where S is the sequence length of the
            %   output. If the number of observations in X is one, scores
            %   will be a K-by-S numeric array.
            %
            %   [net, Y] = predictAndUpdateState(net, X, 'PARAM1', VAL1,
            %   ...) computes predictions with the following optional
            %   name/value pairs:
            %
            %       'MiniBatchSize'     - The size of the mini-batches for
            %                             computing predictions. Larger
            %                             mini-batch sizes lead to faster
            %                             predictions, at the cost of more
            %                             memory. The default is 128.
            %
            %       'ExecutionEnvironment'
            %                           - The execution environment for the
            %                             network. This determines what
            %                             hardware resources will be used
            %                             to run the network.
            %                               - 'auto' - Use a GPU if it is
            %                                 available, otherwise use the
            %                                 CPU.
            %                               - 'gpu' - Use the GPU. To use a
            %                                 GPU, you must have Parallel
            %                                 Computing Toolbox(TM), and a
            %                                 CUDA-enabled NVIDIA GPU with
            %                                 compute capability 3.0 or
            %                                 higher. If a suitable GPU is
            %                                 not available, predict
            %                                 returns an error message.
            %                               - 'cpu' - Use the CPU.
            %                             The default is 'auto'.
            %
            %       'SequenceLength'
            %                           - Strategy to determine the length
            %                           of the sequences used per
            %                           mini-batch. Options are:
            %                               - 'longest' to pad all
            %                               sequences in a batch to the
            %                               length of the longest sequence.
            %                               - 'shortest' to truncate all
            %                               sequences in a batch to the
            %                               length of the shortest
            %                               sequence.
            %                               -  Positive integer - Pad
            %                               sequences to the have same
            %                               length as the longest sequence,
            %                               then split into smaller
            %                               sequences of the specified
            %                               length. If splitting occurs,
            %                               then the function creates extra
            %                               mini-batches.
            %                           The default is 'longest'.
            %
            %       'SequencePaddingValue'
            %                           - Scalar value used to pad
            %                           sequences where necessary. The
            %                           default is 0.
            %
            %   See also SeriesNetwork/classifyAndUpdateState.
            
            % This method is only available for RNNs
            if ~this.IsRNN
                error(message('nnet_cnn:SeriesNetwork:InvalidNetworkForPredictAndUpdateState'));
            end
            
            % Set desired precision
            precision = nnet.internal.cnn.util.Precision('single');
            
            [miniBatchSize, executionEnvironment, sequenceLength, sequencePaddingValue] = iParseAndValidatePredictInputs( varargin{:} );
            if this.PredictEnvironment.isEmpty || ...
                    ~strcmp(this.PredictEnvironment.Value.ExecutionEnvironment, executionEnvironment)
                % If the cache is empty or the user is using another execution
                % environment, then fill the cache
                predictEnvironment = this.getPredictionEnvironment(executionEnvironment);
                this.PredictEnvironment.fillCache(predictEnvironment);
            end
            
            % Get the environment from the cache
            predictEnvironment = this.PredictEnvironment.Value;
            predictNetwork = predictEnvironment.PredictNetwork;
            GPUShouldBeUsed = predictEnvironment.GPUShouldBeUsed;
            
            [Y, finalState, predictNetwork] = this.predictRNN(data, ...
                miniBatchSize, precision, sequenceLength, ...
                sequencePaddingValue, predictNetwork, GPUShouldBeUsed);
            
            % Update the network to the final state
            finalState = iGatherFinalState( finalState, this.StatefulLayers );
            predictNetwork = predictNetwork.updateNetworkState(finalState, this.StatefulLayers);
            this.PrivateNetwork = predictNetwork;
            % Create a copy of the PredictEnvironment cache handle
            predictEnvironment.PredictNetwork = predictNetwork;
            newPredictEnvironmentHandle = nnet.internal.cnn.layer.learnable.CacheHandle();
            newPredictEnvironmentHandle.fillCache( predictEnvironment );
            this.PredictEnvironment = newPredictEnvironmentHandle;
        end
        
        function [labels, scores] = classify(this, X, varargin)
            % classify   Classify data with a network
            %
            %   [labels, scores] = classify(net, X) will classify the data
            %   X using the network net. labels will be an N-by-1
            %   categorical vector where N is the number of observations,
            %   and scores will be an N-by-K matrix where K is the number
            %   of output classes. If the output of the network is a
            %   sequence, labels and scores will be N-by-1 cell arrays.
            %   Each element of labels will be a 1-by-S categorical array
            %   and each element of scores will be a K-by-S numeric array,
            %   where S is the sequence length of the output. The format of
            %   X will depend on the input layer for the network.
            %
            %   For an image input layer, X may be:
            %       - A single image.
            %       - A four dimensional numeric array of images, where the
            %         first three dimensions index the height, width, and
            %         channels of an image, and the fourth dimension
            %         indexes the individual images.
            %       - An image datastore.
            %       - A matlab.io.datastore.MiniBatchDatastore.
            %       - A table, where the first column contains either image
            %         paths or images.
            %
            %   For a network with a sequence input layer, X may be:
            %       - A single time series represented by a numeric array
            %       of size D-by-S, where D is the number of data points
            %       per timestep, and S is the total number of timesteps.
            %       - A cell array of multiple time series, of size N-by-1
            %       where N is the number of observations, and each element
            %       of the cell array contains a D-by-S time series.
            %
            %   [labels, scores] = classify(net, X, 'PARAM1', VAL1, ...)
            %   specifies optional name-value pairs described below:
            %
            %       'MiniBatchSize'     - The size of the mini-batches for
            %                             computing predictions. Larger
            %                             mini-batch sizes lead to faster
            %                             predictions, at the cost of more
            %                             memory. The default is 128.
            %
            %       'ExecutionEnvironment'
            %                           - The execution environment for the
            %                             network. This determines what
            %                             hardware resources will be used
            %                             to run the network.
            %                               - 'auto' - Use a GPU if it is
            %                                 available, otherwise use the
            %                                 CPU.
            %                               - 'gpu' - Use the GPU. To use a
            %                                 GPU, you must have Parallel
            %                                 Computing Toolbox(TM), and a
            %                                 CUDA-enabled NVIDIA GPU with
            %                                 compute capability 3.0 or
            %                                 higher. If a suitable GPU is
            %                                 not available, classify
            %                                 returns an error message.
            %                               - 'cpu' - Use the CPU.
            %                             The default is 'auto'.
            %
            %       'SequenceLength'
            %                           - Strategy to determine the length
            %                           of the sequences used per
            %                           mini-batch. Options are:
            %                               - 'longest' to pad all
            %                               sequences in a batch to the
            %                               length of the longest sequence.
            %                               - 'shortest' to truncate all
            %                               sequences in a batch to the
            %                               length of the shortest
            %                               sequence.
            %                               -  Positive integer - Pad
            %                               sequences to the have same
            %                               length as the longest sequence,
            %                               then split into smaller
            %                               sequences of the specified
            %                               length. If splitting occurs,
            %                               then the function creates extra
            %                               mini-batches.
            %                           The default is 'longest'.
            %
            %       'SequencePaddingValue'
            %                           - Scalar value used to pad
            %                           sequences where necessary. The
            %                           default is 0.
            %
            %   See also SeriesNetwork/predict, SeriesNetwork/activations.
            iAssertIsAClassificationNetworkForClassify(this.IsClassificationNetwork);
            scores = this.predict( X, varargin{:} );
            if ~this.ReturnSequence
                labels = iUndummify( scores, this.Categories );
            else
                labels = iUndummifySequence( scores, this.Categories );
            end
        end
        
        function [this, labels, scores] = classifyAndUpdateState(this, X, varargin)
            % classifyAndUpdateState   Classify data with a network and
            % update the network state
            %
            %   [net, labels, scores] = classifyAndUpdateState(net, X)
            %   classifies the data X using the network net, and updates
            %   its state. labels is an N-by-1 categorical vector where N
            %   is the number of observations, and scores is an N-by-K
            %   matrix where K is the number of output classes. If the
            %   output of the network is a sequence, labels and scores are
            %   N-by-1 cell arrays. Each element of labels is a 1-by-S
            %   categorical array and each element of scores is a K-by-S
            %   numeric array, where S is the sequence length of the
            %   output.
            %
            %   This method only supports networks with recurrent layers.
            %
            %   The format of X may be:
            %       - A single time series represented by a numeric array
            %       of size D-by-S, where D is the number of data points
            %       per timestep, and S is the total number of timesteps.
            %       - A cell array of multiple time series, of size N-by-1
            %       where N is the number of observations, and each element
            %       of the cell array contains a D-by-S time series.
            %
            %   [labels, scores] = classifyAndUpdateState(net, X, 'PARAM1',
            %   VAL1, ...) specifies optional name-value pairs described
            %   below:
            %
            %       'MiniBatchSize'     - The size of the mini-batches for
            %                             computing predictions. Larger
            %                             mini-batch sizes lead to faster
            %                             predictions, at the cost of more
            %                             memory. The default is 128.
            %
            %       'ExecutionEnvironment'
            %                           - The execution environment for the
            %                             network. This determines what
            %                             hardware resources will be used
            %                             to run the network.
            %                               - 'auto' - Use a GPU if it is
            %                                 available, otherwise use the
            %                                 CPU.
            %                               - 'gpu' - Use the GPU. To use a
            %                                 GPU, you must have Parallel
            %                                 Computing Toolbox(TM), and a
            %                                 CUDA-enabled NVIDIA GPU with
            %                                 compute capability 3.0 or
            %                                 higher. If a suitable GPU is
            %                                 not available, classify
            %                                 returns an error message.
            %                               - 'cpu' - Use the CPU.
            %                             The default is 'auto'.
            %
            %       'SequenceLength'
            %                           - Strategy to determine the length
            %                           of the sequences used per
            %                           mini-batch. Options are:
            %                               - 'longest' to pad all
            %                               sequences in a batch to the
            %                               length of the longest sequence.
            %                               - 'shortest' to truncate all
            %                               sequences in a batch to the
            %                               length of the shortest
            %                               sequence.
            %                               -  Positive integer - Pad
            %                               sequences to the have same
            %                               length as the longest sequence,
            %                               then split into smaller
            %                               sequences of the specified
            %                               length. If splitting occurs,
            %                               then the function creates extra
            %                               mini-batches.
            %                           The default is 'longest'.
            %
            %       'SequencePaddingValue'
            %                           - Scalar value used to pad
            %                           sequences where necessary. The
            %                           default is 0.
            %
            %   See also SeriesNetwork/predictAndUpdateState.
            iAssertIsAClassificationNetworkForClassify(this.IsClassificationNetwork);
            [this, scores] = this.predictAndUpdateState( X, varargin{:} );
            if ~this.ReturnSequence
                labels = iUndummify( scores, this.Categories );
            else
                labels = iUndummifySequence( scores, this.Categories );
            end
        end
        
        function Y = activations(this, X, layerID, varargin)
            % activations   Computes network layer activations
            %
            %   Y = activations(net, X, layer) returns network
            %   activations for a specific layer. Network activations are
            %   computed by forward propagating input X through the network
            %   up to the specified layer. layer must be a numeric index
            %   or a character vector corresponding to one of the network
            %   layer names.
            %
            %   For a network with an image input layer, X may be:
            %       - A single image.
            %       - A 4D array of images, where the first three
            %         dimensions index the height, width and channels of an
            %         image, and the fourth dimension indexes the
            %         individual images.
            %       - An image datastore.
            %       - A matlab.io.datastore.MiniBatchDatastore.
            %       - A table, where the first column contains either image
            %         paths or images.
            %
            %   By default for networks with an image input layer, Y will
            %   be an H-by-W-by-C-by-N array, where H, W and C are the
            %   height, width and number of channels for the chosen output.
            %   Each H-by-W-by-C sub-array is the output for a single
            %   observation. See the 'OutputAs' optional name-value pair 
            %   for more details.
            %
            %   The activations function is not supported for networks with
            %   a sequence input layer.
            %
            %   Y = activations(net, X, layer, 'PARAM1', VAL1, ...)
            %   specifies optional name-value pairs described below:
            %
            %       'OutputAs'    - Format of output activations, specified
            %                       as one of the following: 
            %                         - 'channels' - The output is an
            %                           H-by-W-by-C-by-N array, where H, W,
            %                           and C are the height, width and 
            %                           number of channels for the output 
            %                           of the chosen layer. Each 
            %                           H-by-W-by-C sub-array is the output
            %                           for a single observation.
            %                         - 'rows' - The output is an 
            %                           N-by-M matrix, where N is the 
            %                           number of observations, and M is
            %                           the number of output elements in 
            %                           the chosen layer. Each row of the 
            %                           matrix is the output for a single 
            %                           observation.
            %                         - 'columns' - The output is an 
            %                           M-by-N matrix, where M is the 
            %                           number of output elements from the
            %                           chosen layer, and N is the number 
            %                           of observations. Each column of the
            %                           matrix is the output for a single 
            %                           observation.
            %                       The default is 'channels'.
            %
            %       'MiniBatchSize'
            %                     - The size of the mini-batches for
            %                       computing predictions. Larger
            %                       mini-batch sizes lead to faster
            %                       predictions, at the cost of more
            %                       memory. The default is 128.
            %
            %       'ExecutionEnvironment'
            %                     - The execution environment for the
            %                       network. This determines what hardware
            %                       resources will be used to run the
            %                       network.
            %                         - 'auto' - Use a GPU if it is
            %                           available, otherwise use the CPU.
            %                         - 'gpu' - Use the GPU. To use a
            %                           GPU, you must have Parallel
            %                           Computing Toolbox(TM), and a
            %                           CUDA-enabled NVIDIA GPU with
            %                           compute capability 3.0 or higher.
            %                           If a suitable GPU is not available,
            %                           activations returns an error
            %                           message.
            %                         - 'cpu' - Use the CPU.
            %                       The default is 'auto'.
            %
            %   See also SeriesNetwork/predict, SeriesNetwork/classify.
            
            iAssertActivationsInvalidForRNN( this.IsRNN );
            
            % Set desired precision
            precision = nnet.internal.cnn.util.Precision('single');
            
            internalLayers = this.PrivateNetwork.Layers;
            
            layerID = nnet.internal.cnn.util.validateNetworkLayerNameOrIndex(layerID, internalLayers, 'activations');
            
            [miniBatchSize, outputAs, executionEnvironment, sequenceLength, sequencePaddingValue] = ...
                iParseAndValidateActivationsInputs( varargin{:} );
            
            if this.PredictEnvironment.isEmpty || ...
                    ~strcmp(this.PredictEnvironment.Value.ExecutionEnvironment, executionEnvironment)
                % If the cache is empty or the user is using another execution
                % environment, then fill the cache
                predictEnvironment = this.getPredictionEnvironment(executionEnvironment);
                this.PredictEnvironment.fillCache(predictEnvironment);
            end
            
            % Get the environment from the cache
            predictEnvironment = this.PredictEnvironment.Value;
            predictNetwork = predictEnvironment.PredictNetwork;
            GPUShouldBeUsed = predictEnvironment.GPUShouldBeUsed;
            
            singleNumericObservation = isnumeric(X) && (ndims(X) < 4);
            if singleNumericObservation
                
                iAssertInputDataIsValidForActivations( X, this.InputSize, outputAs );
                
                inputSize = iInputSize( X );
                
                % Determine the output size using the private network,
                % which has not been converted to the current predict
                % environment
                outputSize = iDetermineLayerOutputSize( this.PrivateNetwork.Layers, layerID, inputSize );
                
                [sz, indexFcn, reshapeFcn] = iGetOutputSizeAndIndices(...
                    outputAs, 1, outputSize);
                
                % pre-allocate output buffer
                Y = precision.zeros(sz);
                
                data = precision.cast(X);
                YChannelFormat = iActivationsChannelFormatSingleNumericObserv(predictNetwork,data,layerID,GPUShouldBeUsed);
                
                observationIndex = 1;
                indices = indexFcn(observationIndex);
                Y(indices{:}) = reshapeFcn(YChannelFormat, numel(observationIndex));
                
            else
                dispatcher = iDataDispatcher( X, miniBatchSize, precision, ...
                    sequenceLength, sequencePaddingValue, this.Layers );
                
                iAssertInputDataIsValidForActivations( dispatcher, this.InputSize, outputAs );
                
                inputSize = iInputSize( dispatcher );
                
                % Determine the output size using the private network,
                % which has not been converted to the current predict
                % environment
                outputSize = iDetermineLayerOutputSize( this.PrivateNetwork.Layers, layerID, inputSize );
                
                [sz, indexFcn, reshapeFcn] = iGetOutputSizeAndIndices(...
                    outputAs, dispatcher.NumObservations, outputSize);
                
                % pre-allocate output buffer
                Y = precision.zeros(sz);
                
                % Use the dispatcher to run the network on the data
                dispatcher.start();
                while ~dispatcher.IsDone
                    [X, ~, i] = dispatcher.next();
                    
                    YChannelFormat = iActivationsChannelFormatSingleNumericObserv(predictNetwork,X,layerID,GPUShouldBeUsed);
                    
                    indices = indexFcn(i);
                    
                    Y(indices{:}) = reshapeFcn(YChannelFormat, numel(i));
                end
            end
        end
        
        function this = resetState( this )
            % resetState   Reset the state of a recurrent neural network
            %
            %  net = resetState(net) returns the state of a recurrent
            %  neural network to an initial state of zeros.
            
            this.PrivateNetwork = this.PrivateNetwork.resetNetworkState( this.StatefulLayers );
            
            % Clear the network cache
            this.PredictEnvironment.clearCache();
        end
        
        function out = saveobj(this)
            out.Version = 1.0;
            out.Layers = this.Layers; % User visible layers
        end
    end
    
    methods(Static)
        function this = loadobj(in)
            this = SeriesNetwork( in.Layers );
        end
    end
    
    methods (Access=private)
        
        function predictEnvironment = getPredictionEnvironment(this, executionEnvironment)
            % Prepare the network for the correct prediction mode
            gpuShouldBeUsedForPredict = nnet.internal.cnn.util.GPUShouldBeUsed(executionEnvironment);
            if(gpuShouldBeUsedForPredict)
                privateNetwork = this.PrivateNetwork.setupNetworkForGPUPrediction();
            else
                privateNetwork = this.PrivateNetwork.setupNetworkForHostPrediction();
            end
            
            predictEnvironment.PredictNetwork = privateNetwork;
            predictEnvironment.ExecutionEnvironment = executionEnvironment;
            predictEnvironment.GPUShouldBeUsed = gpuShouldBeUsedForPredict;
        end
        
        function [Y, finalState, predictNetwork] = predictRNN(this, data, ...
                miniBatchSize, precision, sequenceLength, ...
                sequencePaddingValue, predictNetwork, GPUShouldBeUsed)
            
            % Create sequence dispatcher
            dispatcher = iDataDispatcher(data, miniBatchSize, precision, ...
                sequenceLength, sequencePaddingValue, this.Layers);
            
            % Validate input data
            dataValidator = nnet.internal.cnn.util.NetworkDataValidator( ...
                nnet.internal.cnn.util.PredictionDataErrorThrower );
            dataValidator.validateDataForPredict(data, dispatcher, this.Layers, this.IsRNN);
            
            % Check initial state
            iAssertInitialStateIsValidForPredict(this.PrivateNetwork.Layers, this.StatefulLayers, dispatcher.MiniBatchSize)
            
            % Inference
            if ~this.ReturnSequence
                Y = precision.zeros([dispatcher.NumObservations this.OutputSize]);
                dispatcher.start();
                while ~dispatcher.IsDone
                    [X, ~, i] = dispatcher.next();
                    
                    if(GPUShouldBeUsed)
                        X = gpuArray(X);
                    end
                    
                    propagateState = dispatcher.IsNextMiniBatchSameObs;
                    [P, state, finalState] = predictNetwork.statefulPredict( X, this.StatefulLayers, propagateState );
                    predictNetwork = predictNetwork.updateNetworkState(state, this.StatefulLayers);
                    Y(i, :) = gather(P)';
                end
            else
                Y = iInitializeSeq2SeqOutput( data, dispatcher.NumObservations );
                dispatcher.start();
                while ~dispatcher.IsDone
                    [X, ~, i] = dispatcher.next();
                    
                    if(GPUShouldBeUsed)
                        X = gpuArray(X);
                    end
                    
                    propagateState = dispatcher.IsNextMiniBatchSameObs;
                    [P, state, finalState] = predictNetwork.statefulPredict( X, this.StatefulLayers, propagateState );
                    predictNetwork = predictNetwork.updateNetworkState(state, this.StatefulLayers);
                    Y = iConvertToSequenceOutput( data, Y, gather(P), i );
                end
            end
        end
        
    end
    
end


function Y = iPredictSingleNumericObserv(predictNetwork,data,GPUShouldBeUsed)

if (GPUShouldBeUsed)
    X = gpuArray(data);
else
    X = data;
end

Y = gather(predictNetwork.predict(X));

end

function YChannelFormat = iActivationsChannelFormatSingleNumericObserv(network,data,layerID,GPUShouldBeUsed)

if GPUShouldBeUsed
    X = gpuArray(data);
else
    X = data;
end

YChannelFormat = gather(network.activations(X, layerID));

end

function iAssertIsAClassificationNetworkForClassify(isClassificationNetwork)
if ~isClassificationNetwork
    exception = iCreateExceptionFromErrorID('nnet_cnn:SeriesNetwork:InvalidNetworkForClassify');
    throwAsCaller(exception);
end
end

function tf = iIsAClassificationNetwork(internalLayers)
tf = isa(internalLayers{end}, 'nnet.internal.cnn.layer.ClassificationLayer');
end

function outputSize = iDetermineLayerOutputSize(layers, layerIdx, inputSize)
% Determine output size of output layer.
if nargin<3
    inputSize = layers{1}.InputSize;
end
for i = 2:layerIdx
    inputSize = layers{i}.forwardPropagateSize(inputSize);
end
outputSize = inputSize;
end

function cats = iCategories( net )
outputLayer = net.Layers{end};
% A classification output layer has to have a property called Categories.
if isprop(outputLayer, 'Categories')
    cats = net.Layers{end}.Categories;
else    
    cats = categorical();
end
end

function labels = iUndummify( scores, categories )
labels = nnet.internal.cnn.util.undummify( scores, categories );
end

function labels = iUndummifySequence( scores, classNames )
isSingleObs = isnumeric( scores );
if isSingleObs
    scores = { scores };
end
labels = cell(numel(scores), 1);
for ii = 1:numel( scores )
    labels{ii} = nnet.internal.cnn.util.undummify( scores{ii}', classNames )';
end
if isSingleObs
    labels = labels{1};
end
end


function iAssertInputDataIsValidForPredict(dispatcher, inputSize, isRNN)
if iDispatchesInputOfSize(dispatcher, inputSize)
elseif isRNN
    exception = iCreateExceptionFromErrorID('nnet_cnn:SeriesNetwork:InvalidPredictDataForSequenceInputLayer');
    throwAsCaller(exception);
else
    exception = iCreateExceptionFromErrorID('nnet_cnn:SeriesNetwork:InvalidPredictDataForImageInputLayer');
    throwAsCaller(exception);
end
end

function iAssertInitialStateIsValidForPredict(layers, statefulLayers, miniBatchSize)
% The state must be specified as a cell array of size numLayers-by-1.
% Its entries must be empty for non-recurrent layers. For recurrent layers,
% entries must be cell arrays of the number of states for the layer. Eg,
% for LSTM, the cell array must have two elements. The states themselves
% must have size hiddenSize-by-1 or hiddenSize-by-miniBatchSize.
if iLayerStatesHaveWrongSize(layers, statefulLayers, miniBatchSize)
    layers = layers( statefulLayers );
    firstStatefulLayer = layers{1};
    stateSize = size( firstStatefulLayer.DynamicParameters(1).Value, 2 );
    error(message('nnet_cnn:SeriesNetwork:IncorrectNetworkState', stateSize, miniBatchSize));
end
end

function tf = iLayerStatesHaveWrongSize(layers, statefulLayers, miniBatchSize)
layers = layers( statefulLayers );
numLayers = numel( layers );
tfVector = true( numLayers, 1 );
for ii = 1:numLayers
    thisLayer = layers{ii};
    numStates = numel( thisLayer.DynamicParameters );
    tfs = true( numStates, 1 );
    for jj = 1:numStates
        tfs(jj) = isequal( size( thisLayer.DynamicParameters(jj).Value, 2 ), 1 ) || ...
            isequal( size( thisLayer.DynamicParameters(jj).Value, 2 ), miniBatchSize );
    end
    tfVector(ii) = all( tfs );
end
tf = ~all( tfVector );
end

function iAssertInputDataIsValidForActivations( dispatcher, inputSize, outputAs )
% iAssertInputDataIsValidForActivations   Throws an error if the dispatcher doesn't given input that
% will work with the 'activations' method.

if iDispatchesImagesSmallerThan(dispatcher, inputSize)
    throwAsCaller( iCreateExceptionFromErrorID( 'nnet_cnn:SeriesNetwork:SmallerImagesForActivations', ...
        mat2str( inputSize ) ) );
    
elseif iDispatchesInputOfSize(dispatcher, inputSize)
    % The 'activations' method can use this data
    
elseif iLargerImagesCanBeOutputAs( outputAs )
    % The 'activations' method can use this data
else
    throwAsCaller( iCreateExceptionFromErrorID( 'nnet_cnn:SeriesNetwork:ActivationsLargeImagesOutputAs' ) );
end
end

function tf = iLargerImagesCanBeOutputAs( outputAs )
tf = isequal(outputAs, 'channels');
end

function iAssertMiniBatchSizeIsValid(x)
if(iIsPositiveIntegerScalar(x))
else
    exception = iCreateExceptionFromErrorID('nnet_cnn:SeriesNetwork:InvalidMiniBatchSize');
    throwAsCaller(exception);
end
end

function sz = iInputSize( x )
% iInputSize   Return the size of x as [H W C] where C is 1 when x is a
% grayscale image.
if iIsDataDispatcher( x )
    sz = x.ImageSize;
    if isempty(sz) && isprop(x, 'DataSize')
        sz = x.DataSize;
    end
else
    sz = [size(x,1) size(x,2) size(x,3)];
end
end

function tf = iDispatchesInputOfSize(dispatcher, inputSize)
dispatchedInputSize = iInputSize( dispatcher );
tf = isequal( dispatchedInputSize, inputSize );
end

function tf = iDispatchesImagesSmallerThan(dispatcher, inputSize)
% iDispatchesImagesSmallerThan   Check if the dispatcher dispatches images
% smaller than inputSize on either one of the first two dimensions or with
% different number of channels.
dispatchedImageSize = iInputSize( dispatcher );
tf = dispatchedImageSize(1) < inputSize(1) || ...
    dispatchedImageSize(2) < inputSize(2) || ...
    dispatchedImageSize(3) ~= inputSize(3);
end

function tf = iIsDataDispatcher(x)
tf = isa(x,'nnet.internal.cnn.DataDispatcher');
end

function tf = iIsPositiveIntegerScalar(x)
tf = all(x > 0) && iIsInteger(x) && isscalar(x);
end

function tf = iIsInteger(x)
tf = isreal(x) && isnumeric(x) && all(mod(x,1)==0);
end

function exception = iCreateExceptionFromErrorID(errorID, varargin)
exception = MException(errorID, getString(message(errorID, varargin{:})));
end

function dispatcher = iDataDispatcher(data, miniBatchSize, precision, sequenceLength, ...
    sequencePaddingValue, layers)
% iDataDispatcher   Use the factory to create a dispatcher.
try
    dispatcher = nnet.internal.cnn.DataDispatcherFactory.createDataDispatcher( ...
        data, [], miniBatchSize, 'truncateLast', precision, ...
        struct( 'useParallel', false ), 'once', sequenceLength, sequencePaddingValue, layers);
catch e
    iThrowInvalidDataException( e, layers(1) )
end
end

function iThrowInvalidDataException(e, inputLayer)
% iThrowInvalidDataException   Throws an InvalidData exception generated
% from DataDispatcherFactory as either an
% InvalidPredictDataForImageInputLayer or
% InvalidPredictDataForSequenceInputLayer exception.
if (strcmp(e.identifier,'nnet_cnn:internal:cnn:DataDispatcherFactory:InvalidData'))
    if isa( inputLayer, 'nnet.cnn.layer.ImageInputLayer' )
        exception = iCreateExceptionFromErrorID('nnet_cnn:SeriesNetwork:InvalidPredictDataForImageInputLayer');
        throwAsCaller(exception)
    elseif isa( inputLayer, 'nnet.cnn.layer.SequenceInputLayer' )
        exception = iCreateExceptionFromErrorID('nnet_cnn:SeriesNetwork:InvalidPredictDataForSequenceInputLayer');
        throwAsCaller(exception)
    end
else
    rethrow(e)
end
end

function [miniBatchSize, executionEnvironment, sequenceLength, sequencePaddingValue] = iParseAndValidatePredictInputs(varargin)

defaultMiniBatchSize = iGetDefaultMiniBatchSize();
defaultExecutionEnvironment = 'auto';
defaultSequenceLength = 'longest';
defaultSequencePaddingValue = 0;

if nargin == 0
    miniBatchSize = defaultMiniBatchSize;
    executionEnvironment = defaultExecutionEnvironment;
    sequenceLength = defaultSequenceLength;
    sequencePaddingValue = defaultSequencePaddingValue;
    return;
end

p = inputParser;

addParameter(p, 'MiniBatchSize', defaultMiniBatchSize);
addParameter(p, 'ExecutionEnvironment', defaultExecutionEnvironment);
addParameter(p, 'SequenceLength', defaultSequenceLength, @(s)iValidateSequenceLength(s));
addParameter(p, 'SequencePaddingValue', defaultSequencePaddingValue, @(x)iValidateSequencePaddingValue(x));

parse(p, varargin{:});
iAssertMiniBatchSizeIsValid(p.Results.MiniBatchSize);

miniBatchSize = p.Results.MiniBatchSize;
executionEnvironment = iValidateExecutionEnvironment(p.Results.ExecutionEnvironment, 'predict');
sequenceLength = p.Results.SequenceLength;
sequencePaddingValue = p.Results.SequencePaddingValue;
end

function [miniBatchSize, outputAs, executionEnvironment, sequenceLength, sequencePaddingValue] = iParseAndValidateActivationsInputs(varargin)
p = inputParser;

defaultMiniBatchSize = iGetDefaultMiniBatchSize();
defaultOutputAs = 'channels';
defaultExecutionEnvironment = 'auto';
defaultSequenceLength = 'longest';
defaultSequencePaddingValue = 0;

addParameter(p, 'MiniBatchSize', defaultMiniBatchSize);
addParameter(p, 'OutputAs', defaultOutputAs);
addParameter(p, 'ExecutionEnvironment', defaultExecutionEnvironment);
addParameter(p, 'SequenceLength', defaultSequenceLength, @(s)iValidateSequenceLength(s));
addParameter(p, 'SequencePaddingValue', defaultSequencePaddingValue, @(x)iValidateSequencePaddingValue(x));
parse(p, varargin{:});

iAssertMiniBatchSizeIsValid(p.Results.MiniBatchSize);

miniBatchSize = p.Results.MiniBatchSize;
outputAs = iValidateOutputAs(p.Results.OutputAs);
executionEnvironment = iValidateExecutionEnvironment(p.Results.ExecutionEnvironment, 'activations');
sequenceLength = p.Results.SequenceLength;
sequencePaddingValue = p.Results.SequencePaddingValue;
end

function iAssertActivationsInvalidForRNN( isRNN )
% Assert that the activations method is not being called on an RNN.
if isRNN
    exception = iCreateExceptionFromErrorID('nnet_cnn:SeriesNetwork:ActivationsInvalidForRNN');
    throwAsCaller(exception);
end
end

function [outputBatchSize, indexFcn, reshapeFcn] = iGetOutputSizeAndIndices(outputAs, numObs, outputSize)
% Returns the output batch size, indexing function, and reshaping function.
% The indexing function provides the right set of indices based on the
% 'OutputAs' setting. The reshaping function reshapes channel
% formatted output to the shape required for the 'OutputAs' setting.
switch outputAs
    case 'rows'
        outputBatchSize = [numObs prod(outputSize)];
        indexFcn = @(i){i 1:prod(outputSize)};
        reshapeFcn = @(y,n)transpose(reshape(y, [], n));
    case 'columns'
        outputBatchSize = [prod(outputSize) numObs];
        indexFcn = @(i){1:prod(outputSize) i};
        reshapeFcn = @(y,n)reshape(y, [], n);
    case 'channels'
        outputBatchSize = [outputSize numObs];
        indices = arrayfun(@(x)1:x, outputSize, 'UniformOutput', false);
        indexFcn = @(i)[indices i];
        reshapeFcn = @(y,~)y;
end
end

function val = iGetDefaultMiniBatchSize
val = 128;
end

function valid = iValidateOutputAs(str)
validChoices = {'rows', 'columns', 'channels'};
valid = validatestring(str, validChoices, 'activations', 'OutputAs');
end

function validString = iValidateExecutionEnvironment(inputString, caller)
validExecutionEnvironments = {'auto', 'gpu', 'cpu'};
validString = validatestring(inputString, validExecutionEnvironments, caller, 'ExecutionEnvironment');
end

function YFormatted = iFormatPredictions(Y)
% iFormatPredictions   Format predictions according to the problem.
% If Y is [1 1 K N], then YFormatted will be [N K].
% If Y is [H W K N], with H or W not singleton, then YFormatted will be the
% same as Y.

YSize = size(Y);
if YSize(1)==1 && YSize(2)==1
    % Get rid of first two singleton dimensions
    YFormatted = shiftdim(Y,2);
    % Transpose [K N] -> [N K]
    YFormatted = YFormatted';
else
    YFormatted = Y;
end
end

function layersMap = iLayersMap( layers )
layersMap = nnet.internal.cnn.layer.util.InternalExternalMap( layers );
end

function state = iGatherFinalState( state, statefulLayerInds )
for ii = 1:numel( state )
    if statefulLayerInds(ii)
        state{ii} = cellfun(@gather, state{ii}, 'UniformOutput', false);
    end
end
end

function Y = iInitializeSeq2SeqOutput( data, numObs )
if iscell( data )
    Y = cell( numObs, 1 );
else
    Y = [];
end
end

function Y = iConvertToSequenceOutput( X, Y, P, inds )
% Sequence output can have one of two forms:
%   i) For a single observation, the output should be a
%   numClasses-by-sequenceLength numeric array
%   ii) For multiple observations, the output should be a numObs-by-1 cell
%   array, containing a numClasses-by-sequenceLength numeric array for each
%   observation.
if isnumeric( X )
    % Single observation
    if iscell( Y )
        Y = Y{1};
    end
    Y = iAppendSequencePredictions( X, Y, P );
else
    % Multiple observations
    for ii = 1:numel(inds)
        Y{ inds(ii) } = iAppendSequencePredictions( X{ inds(ii) }, Y{ inds(ii) }, P(:, ii, :) );
    end
end
end

function Y = iAppendSequencePredictions( X, Y, P )
yIndicesToAppend = size( X, 2 ) - size( Y, 2 );
pIndicesToAppend = min( yIndicesToAppend, size( P, 3 ) );
Y = [Y iPermuteSequenceDimensions( P(:, 1, 1:pIndicesToAppend ) )];
end

function y = iPermuteSequenceDimensions( x )
% Permute sequence dimensions of 3D sequence array, swapping the sequence
% dimension and observation dimension
%   D-by-N-by-S --> D-by-S-by-N
y = permute( x, [1 3 2] );
end

function iValidateSequenceLength( x )
try
    if ischar(x) || isstring(x)
        validatestring(x, {'longest', 'shortest'});
    else
        validateattributes(x, {'numeric'}, {'positive', 'integer', 'real'});
    end
catch
    error(message('nnet_cnn:SeriesNetwork:InvalidSequenceLength'));
end
end

function iValidateSequencePaddingValue( x )
validateattributes(x, {'numeric'}, {'scalar', 'real'} )
end