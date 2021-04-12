classdef Softmax < nnet.internal.cnn.layer.Layer
    % Softmax   Implementation of the softmax layer
    
    %   Copyright 2015-2017 The MathWorks, Inc.
    
    properties
        % LearnableParameters   Learnable parameters for the layer
        %   This layer has no learnable parameters.
        LearnableParameters = nnet.internal.cnn.layer.learnable.PredictionLearnableParameter.empty();

        % Name (char array)   A name for the layer
        Name
        
        % VectorFormat (Boolean)   Format of the layer's input. If true,
        % the layer takes vector inputs (D-by-N-by-S). If false, the layer
        % expects image inputs (H-by-W-by-C-by-N). The default is false.
        VectorFormat = false
    end
    
    properties (Constant)
        % DefaultName   Default layer's name.
        DefaultName = 'softmax'
    end
            
    properties(SetAccess = private)
        % InputNames   This layer has a single input
        InputNames = {'in'}
        
        % OutputNames   This layer has a single output
        OutputNames = {'out'}
        
        % HasSizeDetermined   Specifies if all size parameters are set
        HasSizeDetermined = false
        
        % ExecutionStrategy   The execution strategy for this layer
        ExecutionStrategy
    end
    
    methods
        function this = Softmax(name)
            this.Name = name;
            
            this.ExecutionStrategy = this.getHostStrategy();
        end
        
        function Z = predict(this, X)
            Z = this.ExecutionStrategy.forward(X);
        end
        
        function [dX,dW] = backward(this, ~, Z, dZ, ~)
            [dX,dW] = this.ExecutionStrategy.backward(Z, dZ);
        end
        
        function outputSize = forwardPropagateSize(~, inputSize)
            outputSize = inputSize;
        end
        
        function this = inferSize(this, inputSize)
            if isscalar(inputSize)
                this.VectorFormat = true;
            else
                this.VectorFormat = false;
            end
            this.ExecutionStrategy = this.getHostStrategy;
        end
        
        function tf = isValidInputSize(~, inputSize)
            % isValidInputSize   Check if the layer can accept an input of
            % a certain size
            
            tf = isscalar(inputSize) || ...
                (iNonEmptyMatrix(inputSize(1:2)) && numel(inputSize)<=3);
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
            this.ExecutionStrategy = this.getHostStrategy();
        end
        
        function this = setupForGPUPrediction(this)
            this.ExecutionStrategy = this.getGPUStrategy();
        end
        
        function this = setupForHostTraining(this)
            this.ExecutionStrategy = this.getHostStrategy();
        end
        
        function this = setupForGPUTraining(this)
            this.ExecutionStrategy = this.getGPUStrategy();
        end
        
    end
        
    methods(Access = private)
        function executionStrategy = getHostStrategy(this)
            if this.VectorFormat
                executionStrategy = nnet.internal.cnn.layer.util.SoftmaxHostVectorStrategy();
            else
                executionStrategy = nnet.internal.cnn.layer.util.SoftmaxHostImageStrategy();
            end
        end
        
        function executionStrategy = getGPUStrategy(this)
            if this.VectorFormat
                executionStrategy = nnet.internal.cnn.layer.util.SoftmaxGPUVectorStrategy();
            else
                executionStrategy = nnet.internal.cnn.layer.util.SoftmaxGPUImageStrategy();
            end
        end
    end
end

function tf = iNonEmptyMatrix(inputSize)
tf = all(inputSize > 0);
end