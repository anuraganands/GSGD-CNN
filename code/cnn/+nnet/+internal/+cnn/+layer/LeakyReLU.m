classdef LeakyReLU < nnet.internal.cnn.layer.Layer
    % LeakyReLU   Leaky Rectified Linear Unit layer
    
    %   Copyright 2016-2017 The MathWorks, Inc.
    
    properties
        % LearnableParameters   Learnable parameters for the layer
        %   This layer has no learnable parameters.
        LearnableParameters = nnet.internal.cnn.layer.learnable.PredictionLearnableParameter.empty();

        % Name (char array)   A name for the layer
        Name

        % Scale (numeric scalar) Scaling factor for negative input values
        Scale
    end
    
    properties (Constant)
        % DefaultName   Default layer's name.
        DefaultName = 'leakyrelu'
    end
            
    properties (SetAccess = private)
        % InputNames   This layer has a single input
        InputNames = {'in'}
        
        % OutputNames   This layer has a single output
        OutputNames = {'out'}
        
        % HasSizeDetermined   True for layers with size determined.
        HasSizeDetermined = true
    end
    
    properties(Access = private)
        ExecutionStrategy
    end
    
    methods
        function this = LeakyReLU(name, scale)
            % ReLU  Constructor for the layer
            this.Name = char(name);
            this.Scale = scale;
            
            this.ExecutionStrategy = nnet.internal.cnn.layer.util.LeakyReLUHostStrategy();
        end
        
        function Z = predict( this, X )
            % predict   Forward input data through the layer and output the result
            Z = this.ExecutionStrategy.forward(X, this.Scale);
        end
        
        function [dX,dW] = backward( this, X, Z, dZ, ~ )
            % backward    Back propagate the derivative of the loss function
            % through the layer
            [dX,dW] = this.ExecutionStrategy.backward(Z, dZ, X, this.Scale);
        end
        
        function outputSize = forwardPropagateSize(~, inputSize)
            % forwardPropagateSize  Output the size of the layer based on
            %                       the input size
            outputSize = inputSize;
        end
        
        function this = inferSize(this, ~)
            % inferSize     no-op since this layer has nothing that can be
            %               inferred
        end
        
        function tf = isValidInputSize(~, ~)
            % isValidInputSize   Check if the layer can accept an input of
            % a certain size
            tf = true;
        end
        
        function this = initializeLearnableParameters(this, ~)
            % initializeLearnableParameters     no-op since there are no
            %                                   learnable parameters
        end
        
        function this = prepareForTraining(this)
            this.LearnableParameters = nnet.internal.cnn.layer.learnable.TrainingLearnableParameter.empty();
        end
        
        function this = prepareForPrediction(this)
            this.LearnableParameters = nnet.internal.cnn.layer.learnable.PredictionLearnableParameter.empty();
        end
        
        function this = setupForHostPrediction(this)
            this.ExecutionStrategy = nnet.internal.cnn.layer.util.LeakyReLUHostStrategy();
        end
        
        function this = setupForGPUPrediction(this)
            this.ExecutionStrategy = nnet.internal.cnn.layer.util.LeakyReLUGPUStrategy();
        end
        
        function this = setupForHostTraining(this)
            this.ExecutionStrategy = nnet.internal.cnn.layer.util.LeakyReLUHostStrategy();
        end
        
        function this = setupForGPUTraining(this)
            this.ExecutionStrategy = nnet.internal.cnn.layer.util.LeakyReLUGPUStrategy();
        end
    end
end
