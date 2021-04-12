classdef OptimizeChannelAverage < nnet.internal.cnn.visualize.layer.VisualizationOutputLayer
    % OptimizeChannelAverage   Implementation
    
    %   Copyright 2016-2017 The MathWorks, Inc.
    
    properties
        % LearnableParameters   Learnable parameters for the layer
        %   This layer has no learnable parameters.
        LearnableParameters = nnet.internal.cnn.layer.learnable.PredictionLearnableParameter.empty();

        % Name (char array)   A name for the layer
        Name
    end
    
    properties (Constant)
        % DefaultName   Default layer's name.
        DefaultName = 'optimizechannelaverage'
    end
            
    properties (SetAccess = private)
        % HasSizeDetermined   Specifies if all size parameters are set
        %   For pooling layers, this is always true.
        HasSizeDetermined = true
        
        % ChannelNumbers   An array of channel numbers to optimize for
        Channels
    end
    
    methods
        function this = OptimizeChannelAverage(channels)
            this.Name = nnet.internal.cnn.visualize.layer.OptimizeChannelAverage.DefaultName;
            
            % Set channels
            this.Channels = channels;
        end
        
        function Z = predict(~, ~) %#ok<STOUT>
            error(message('nnet_cnn:internal:cnn:visualize:layer:OptimizeChannelAverage:PredictProhibited'));
        end
        
        function [Z, memory] = forward(~, ~) %#ok<STOUT>
            error(message('nnet_cnn:internal:cnn:visualize:layer:OptimizeChannelAverage:ForwardProhibited'));
        end
        
        function dX = backward( ~, ~, ~, ~, ~ ) %#ok<STOUT>
            error(message('nnet_cnn:internal:cnn:visualize:layer:OptimizeChannelAverage:BackwardProhibited'));
        end
        
        function Z = forwardActivations(this, X)
            numChannels = size(this.Channels, 2);
            Z = zeros([1, numChannels], 'like', X);
            for i=1:numChannels
                channelOutput = X(:,:,this.Channels(i),i);
                Z(i) = mean(mean(channelOutput));
            end
        end
        
        function dX = backwardActivations(this, X, Z)
            dX = zeros(size(X), 'like', X);
            numChannels = size(this.Channels, 2);
            for i = 1:numChannels
                channelOutput = Z(i);
                dX(:,:,this.Channels(i),i) = ...
                    ones(size(channelOutput)) / (size(channelOutput, 1)*size(channelOutput, 2));
            end
        end
        
        function gradients = gradients( ~, ~, ~ )
            % gradients    No-op since this layer does not contain any
            % learnable parameters
            gradients = [];
        end

        function outputSize = forwardPropagateSize(this, ~)
            outputSize = [1,1,1,size(this.Channels,2)];
        end

        function this = inferSize(this, ~)
        end
        
        function tf = isValidInputSize(~, ~) %#ok<STOUT>
            error(message('nnet_cnn:internal:cnn:visualize:layer:OptimizeChannelAverage:IsValidInputSizeProhibited'));
        end     
        
        function this = initializeLearnableParameters(this, ~)
        end
        
        function this = prepareForTraining(this)
            this.LearnableParameters = nnet.internal.cnn.layer.learnable.TrainingLearnableParameter.empty();
        end
        
        function this = prepareForPrediction(this)
            this.LearnableParameters = nnet.internal.cnn.layer.learnable.PredictionLearnableParameter.empty();
        end
        
        function this = setupForHostPrediction(this)
        end
        
        function this = setupForGPUPrediction(this)
        end
        
        function this = setupForGPUTraining(this)
        end
        
        function this = setupForHostTraining(this)
        end
    end
end