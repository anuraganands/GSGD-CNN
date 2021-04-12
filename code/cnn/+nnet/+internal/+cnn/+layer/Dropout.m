classdef Dropout < nnet.internal.cnn.layer.Layer
    % Dropout   Implementation of the dropout layer
    
    %   Copyright 2015-2017 The MathWorks, Inc.
    
    properties
        % LearnableParameters   Learnable parameters for the layer
        %   This layer has no learnable parameters.
        LearnableParameters = nnet.internal.cnn.layer.learnable.PredictionLearnableParameter.empty();
        
        % Name (char array)   A name for the layer
        Name
    end
    
    properties (Constant)
        % DefaultName   Default layer's name.
        DefaultName = 'dropout'
    end
    
    properties (SetAccess = private)
        % InputNames   This layer has a single input
        InputNames = {'in'}
        
        % OutputNames   This layer has a single output
        OutputNames = {'out'}
        
        % HasSizeDetermined   Specifies if all size parameters are set
        %   For this layer, there are no size parameters to set.
        HasSizeDetermined = true
        
        % Fraction   The proportion of neurons to drop
        %   A number between 0 and 1 which specifies the proportion of
        %   input elements that are dropped by the dropout layer.
        Probability
    end
    
    methods
        function this = Dropout(name, probability)
            this.Name = name;
            this.Probability = probability;
        end
        
        function Z = predict(~, X)
            Z = X;
        end
        
        function [Z, dropoutMask] = forward(this, X)
            % Use "inverted dropout", where we use scaling at training time
            % so that we don't have to scale at test time. The scaled
            % dropout mask is returned as the variable "dropoutMask".
            dropoutScaleFactor = cast( 1 - this.Probability, superiorfloat(X) );
            dropoutMask = ( rand(size(X), 'like', X) > this.Probability ) / dropoutScaleFactor;
            Z = X.*dropoutMask;
        end
        
        function [dX,dW] = backward(~, ~, ~, dZ, mask)
            dX = dZ.*mask;
            dW = []; % No learnable parameters
        end
        
        function outputSize = forwardPropagateSize(~, inputSize)
            outputSize = inputSize;
        end
        
        function this = inferSize(this, ~)
        end
        
        function tf = isValidInputSize(~, ~)
            % isValidInputSize   Check if the layer can accept an input of
            % a certain size
            tf = true;
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
        
        function this = setupForHostTraining(this)
        end
        
        function this = setupForGPUTraining(this)
        end
    end
end