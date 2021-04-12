classdef Addition < nnet.internal.cnn.layer.Layer
    % Addition  Addition layer.
    
    %   Copyright 2017 The MathWorks, Inc.
    
    properties
        % LearnableParameters   Learnable parameters for the layer
        %   This layer has no learnable parameters.
        LearnableParameters = nnet.internal.cnn.layer.learnable.PredictionLearnableParameter.empty();

        % Name (char array)   A name for the layer
        Name
        
        % NumInputs
        NumInputs
    end
    
    properties (Constant)
        % DefaultName   Layer's default name
        DefaultName = 'addition'
    end
    
    properties(SetAccess = private, Dependent)
        % InputNames   The names are determined by the number of inputs
        InputNames
    end
    
    properties (SetAccess = private)
        % OutputNames   There is one output for this layer
        OutputNames = {'out'}
        
        % HasSizeDetermined
        HasSizeDetermined = true
    end
    
    methods
        function this = Addition(name, numInputs)
            % Addition   Constructor for this layer
            this.Name = name;
            this.NumInputs = numInputs;
        end
        
        function Z = predict( this, X )
            Z = X{1};
            for i = 2:this.NumInputs
                Z = Z + X{i};
            end
        end
        
        function [dX, dW] = backward( this, ~, ~, dZ, ~ )
            dX = cell(1,this.NumInputs);
            for i = 1:this.NumInputs
                dX{i} = dZ;
            end            
            
            % There are no learnable parameters.
            dW = [];
        end
        
        function gradients = gradients( ~, ~, ~ )
            % gradients    No-op since this layer does not contain any
            % learnable parameters
            gradients = [];
        end
        
        function outputSize = forwardPropagateSize(~, inputSize)
            inputSize = iWrapInCell(inputSize);
            inputHeight = inputSize{1}(1);
            inputWidth = inputSize{1}(2);
            totalNumChannels = inputSize{1}(3);
            outputSize = [inputHeight inputWidth totalNumChannels];
        end
        
        function this = inferSize(this, ~)
        end
        
        function tf = isValidInputSize(this, inputSize)
            tf = iscell(inputSize) && ...
                iNumInputsMatchLayersInputs(inputSize, this.NumInputs) && ...
                iSpecifiedDimensionIsEqualForAllInputs(inputSize, 1) && ...
                iSpecifiedDimensionIsEqualForAllInputs(inputSize, 2) && ...
                iSpecifiedDimensionIsEqualForAllInputs(inputSize, 3);
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
        end
        
        function this = setupForGPUPrediction(this)
        end
        
        function this = setupForHostTraining(this)
        end
        
        function this = setupForGPUTraining(this)
        end
        
        function val = get.InputNames(this)
            val = iGenerateInputNames(this.NumInputs);
        end
    end
end

function X = iWrapInCell(X)
if ~iscell(X)
    X = {X};
end
end

function tf = iNumInputsMatchLayersInputs(inputSize, layersNumInputs)
numInputs = numel(inputSize);
tf = numInputs == layersNumInputs;
end

function tf = iSpecifiedDimensionIsEqualForAllInputs(inputSize, dimension)
numInputs = numel(inputSize);
dimensionValues = zeros(1,numInputs);
for i = 1:numInputs
    dimensionValues(i) = inputSize{i}(dimension);
end
tf = iAllElementsAreEqual(dimensionValues);
end

function tf = iAllElementsAreEqual(x)
tf = all(x == x(1));
end

function inputNames = iGenerateInputNames(numInputs)
inputNames = cellstr("in" + (1:numInputs));
end