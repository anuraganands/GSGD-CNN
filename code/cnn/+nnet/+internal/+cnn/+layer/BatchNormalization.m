classdef BatchNormalization < nnet.internal.cnn.layer.Layer ...
        & nnet.internal.cnn.layer.Finalizable
    % BatchNormalization   Implementation of the batch normalization layer
    
    %   Copyright 2017 The MathWorks, Inc.
    
    properties
        % LearnableParameters   Learnable parameters for the layer
        % (Vector of
        % nnet.internal.cnn.layer.learnable.PredictionLearnableParameter)
        LearnableParameters = nnet.internal.cnn.layer.learnable.PredictionLearnableParameter.empty();
        
        % Name  A name for the layer (char vector)
        Name
        % Epsilon  Offset for variance to avoid divide-by-zero (double
        % scalar)
        Epsilon
        
        % NumTrainingSamples  Number of samples used to compute
        % TrainedMean and TrainedVariance.
        NumTrainingSamples = 0;
    end
    
    properties (Constant)
        % DefaultName   Default layer's name.
        DefaultName = 'batchnorm'
    end
    
    properties (SetAccess = private)
        % InputNames   This layer has a single input
        InputNames = {'in'}
        
        % OutputNames   This layer has a single output
        OutputNames = {'out'}
        
        % HasSizeDetermined  True for layers with size determined.
        HasSizeDetermined
        
        % NumChannels  Number of channels in the input. Leave as [] to
        % infer later.
        NumChannels
    end
    
    properties(Access = private)
        ExecutionStrategy
        
        IsTraining
        
        % The trained values should be cached on the GPU during to improve
        % performance. This should be hidden from the caller, which always
        % receives a host-side copy.
        TrainedMeanCache
        TrainedVarianceCache
    end
    
    properties (Dependent)
        % Learnable Parameters (nnet.internal.cnn.layer.LearnableParameter)
        Offset
        Scale
        % TrainedMean  Per-channel mean of the layer input data, determined
        % during training.
        TrainedMean
        % TrainedVariance  Per-channel variance of the layer input data,
        % determined during training.
        TrainedVariance        
    end
    
    properties (Constant, Access = private)
        % OffsetIndex  Index of Offset within LearnableParameters
        OffsetIndex = 1;
        % ScaleIndex  Index of Scale within LearnableParameters
        ScaleIndex = 2;
    end
    
    methods (Access = private)
        function this = initializePopulationStatistics(this)
            this.TrainedMeanCache = nnet.internal.cnn.layer.util.CachedParameter([]);
            this.TrainedVarianceCache = nnet.internal.cnn.layer.util.CachedParameter([]);
            this.NumTrainingSamples = 0;
        end
    end
    
    methods
        function this = BatchNormalization(name, numChannels, Epsilon)
            % BatchNormalization   Constructor for a BatchNormalization layer
            %
            %   Create a batch normalization layer with the following
            %   compulsory parameters:
            %
            %       name        - Name for the layer
            %       numChannels - The number of channels that the input to the
            %                     layer will have. Use [] to leave it unset.
            %       Epsilon     - Scalar specifying variance offset to avoid
            %                     divide-by-zero.
            
            this.Name = name;
            
            % Set Hyper-parameters
            this.NumChannels = numChannels;
            this.HasSizeDetermined = ~isempty( numChannels );
            this.Epsilon = Epsilon;
            
            % Initialize population statistics
            this = initializePopulationStatistics(this);
            
            % Set up LearnableParameters
            this.Offset = nnet.internal.cnn.layer.learnable.PredictionLearnableParameter();
            this.Scale = nnet.internal.cnn.layer.learnable.PredictionLearnableParameter();
            
            this.ExecutionStrategy = nnet.internal.cnn.layer.util.BatchNormalizationHostStrategy();
            this.IsTraining = false;
        end
        
        function [Z, memory] = forward( this, X )
            % forward   Forward input data through the layer at training
            % time and output the result
            [Z, memory] = this.ExecutionStrategy.forwardTrain( ...
                X, ...
                this.Offset.Value, this.Scale.Value, this.Epsilon );
        end
        
        function Z = predict( this, X )
            % predict   Forward input data through the layer at prediction
            % time and output the result
            
            % If being called in training mode, use forward. If in predict
            % then we can expect the finalized mean and variance to be
            % available.
            if this.IsTraining
                Z = this.ExecutionStrategy.forwardTrain( ...
                    X, ...
                    this.Offset.Value, this.Scale.Value, this.Epsilon );
                return
            elseif isempty(this.TrainedMean) || isempty(this.TrainedVariance)
                error( message('nnet_cnn:layer:BatchNormalizationLayer:NotFinalized') );
            end
            
            Z = this.ExecutionStrategy.forwardPredict( ...
                X, ...
                this.Offset.Value, this.Scale.Value, this.Epsilon, ...
                this.TrainedMeanCache.Value, this.TrainedVarianceCache.Value );
        end
        
        function varargout = backward( this, X, Z, dZ, memory )
            % backward    Back propagate the derivative of the loss function
            % through the layer
            [ varargout{1:nargout} ] = this.ExecutionStrategy.backward(Z, dZ, X, ...
                this.Scale.Value, this.Epsilon, memory);
        end

        function layer = finalize(layer, X, ~, memory )
            % Update running estimates of input mean and variance over the
            % training population.
            [batchMean, batchInvStd] = deal(memory{:});
            batchVar = 1./batchInvStd.^2;
            N = numel(X) ./ numel(batchMean); % How many input samples per channel
            
            [newMean, newVar, newN] = iMergeStats( ...
                layer.TrainedMean, layer.TrainedVariance, layer.NumTrainingSamples, ...
                batchMean, batchVar, N);
            
            % Always make sure the final parameters are on the host
            layer.TrainedMean = gather(newMean);
            layer.TrainedVariance = gather(newVar);
            layer.NumTrainingSamples = gather(newN);
        end
        
        function layer = mergeFinalized(layer, layer2)
            % Merge two partially finalized layer objects
            [newMean, newVar, newN] = iMergeStats( ...
                layer.TrainedMean, layer.TrainedVariance, layer.NumTrainingSamples, ...
                layer2.TrainedMean, layer2.TrainedVariance, layer2.NumTrainingSamples);
            
            layer.TrainedMean = newMean;
            layer.TrainedVariance = newVar;
            layer.NumTrainingSamples = newN;
        end
        
        function this = inferSize(this, inputSize)
            % inferSize     Infer the number of channels based on the input size
            if numel(inputSize)<3
                this.NumChannels = 1;
            else
                this.NumChannels = inputSize(3);
            end
            this.HasSizeDetermined = true;
        end
        
        function tf = isValidInputSize(this, inputSize)
            % isValidInputSize   Check if the layer can accept an input of
            % a certain size. Any input with the right number of channels
            % (third dimension) can be accepted.
            if ~this.HasSizeDetermined
                % If we haven't yet specified a size, any size is fine.
                tf = true;
                return
            end
            
            if numel(inputSize)>=3
                N = inputSize(3);
            else
                N = 1;
            end
            tf = (N == this.NumChannels);
        end
        
        function outputSize = forwardPropagateSize(~, inputSize)
            % forwardPropagateSize    Output the size of the layer based on
            % the input size
            outputSize = inputSize;
        end
        
        function this = initializeLearnableParameters(this, precision)
            % initializeLearnableParameters    Initialize learnable
            % parameters using their initializer
            sz = [1 1 this.NumChannels];
            if isempty(this.Offset.Value)
                % Initialize only if it is empty
                this.Offset.Value = precision.zeros(sz);
            else
                % Cast to desired precision
                this.Offset.Value = precision.cast(this.Offset.Value);
            end
            
            if isempty(this.Scale.Value)
                % Initialize only if it is empty
                this.Scale.Value = precision.ones(sz);
            else
                % Cast to desired precision
                this.Scale.Value = precision.cast(this.Scale.Value);
            end
            
            % Also cast the population statistics
            this.TrainedMean = precision.cast(this.TrainedMean);
            this.TrainedVariance = precision.cast(this.TrainedVariance);
        end
        
        function this = prepareForTraining(this)
            % prepareForTraining   Prepare this layer for training
            %   Before this layer can be used for training, we need to
            %   convert the learnable parameters to use the class
            %   TrainingLearnableParameter.
            this.LearnableParameters = nnet.internal.cnn.layer.learnable.convert2training(this.LearnableParameters);
            this.IsTraining = true;
            
            this = initializePopulationStatistics(this);
        end
        
        function this = prepareForPrediction(this)
            % prepareForPrediction   Prepare this layer for prediction
            %   Before this layer can be used for prediction, we need to
            %   convert the learnable parameters to use the class
            %   PredictionLearnableParameter.
            this.LearnableParameters = nnet.internal.cnn.layer.learnable.convert2prediction(this.LearnableParameters);
            this.IsTraining = false;
        end
        
        function this = setupForHostPrediction(this)
            this.ExecutionStrategy = nnet.internal.cnn.layer.util.BatchNormalizationHostStrategy();
            this.LearnableParameters(1).UseGPU = false;
            this.LearnableParameters(2).UseGPU = false;
            this.TrainedMeanCache.UseGPU = false;
            this.TrainedVarianceCache.UseGPU = false;
        end
        
        function this = setupForGPUPrediction(this)
            this.ExecutionStrategy = nnet.internal.cnn.layer.util.BatchNormalizationGPUStrategy();
            this.LearnableParameters(1).UseGPU = true;
            this.LearnableParameters(2).UseGPU = true;
            this.TrainedMeanCache.UseGPU = true;
            this.TrainedVarianceCache.UseGPU = true;
        end
        
        function this = setupForHostTraining(this)
            this.ExecutionStrategy = nnet.internal.cnn.layer.util.BatchNormalizationHostStrategy();
            % Also make sure any trained parameters are gathered
            [this.TrainedMean, this.TrainedVariance] = ...
                gather(this.TrainedMean, this.TrainedVariance);
        end
        
        function this = setupForGPUTraining(this)
            this.ExecutionStrategy = nnet.internal.cnn.layer.util.BatchNormalizationGPUStrategy();
        end
        
        % Setter and getter for learnable parameters.
        % These allow addressing the learnable parameters by name.
        function value = get.Offset(this)
            value = this.LearnableParameters(this.OffsetIndex);
        end
        
        function this = set.Offset(this, value)
            this.LearnableParameters(this.OffsetIndex) = value;
        end
        
        function value = get.Scale(this)
            value = this.LearnableParameters(this.ScaleIndex);
        end
        
        function this = set.Scale(this, value)
            this.LearnableParameters(this.ScaleIndex) = value;
        end
        
        function value = get.TrainedMean(this)
            value = this.TrainedMeanCache.HostValue;
        end
        
        function value = get.TrainedVariance(this)
            value = this.TrainedVarianceCache.HostValue;
        end
        
        function this = set.TrainedMean(this, value)
            this.TrainedMeanCache.Value = value;
        end
        
        function this = set.TrainedVariance(this, value)
            this.TrainedVarianceCache.Value = value;
        end
    end
end


function [mean3, var3, N3] = iMergeStats(mean1, var1, N1, mean2, var2, N2)
% Calculate the combined population mean and variance given two smaller
% population means and variances.

if N1 == 0
    % No samples in first stats, so just use second
    N3 = N2;
    mean3 = mean2;
    var3 = var2;
    
elseif N2 == 0
    % No samples in second stats, so just use first
    N3 = N1;
    mean3 = mean1;
    var3 = var1;
    
else
    % Calculate combined mean and number of samples
    N3 = N1 + N2;
    ratio = N2 ./ N3;
    mean3 = (1-ratio).*mean1 + ratio.*mean2;
    % Use the calculated mean to calculate an updated variance
    var3 = (1-ratio) .* (var1 + mean1.^2) ...
        + ratio .* (var2 + mean2.^2) ...
        - mean3.^2;
end
end