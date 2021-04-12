classdef Concatenation < nnet.internal.cnn.layer.Layer
    % Concatenation   Concatenation layer
    
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
        DefaultName = 'depthcat'
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
        
        % ConcatenationAxis
        ConcatenationAxis
    end
    
    methods
        function this = Concatenation(name, concatenationAxis, numInputs)
            % Concatenation   Constructor for this layer
            this.Name = name;
            this.ConcatenationAxis = concatenationAxis;
            this.NumInputs = numInputs;
        end
        
        function Z = predict( this, X )
            X = iWrapSingleInputInCell(X);
            Z = cat(this.ConcatenationAxis, X{:});
        end
        
        function [dX, dW] = backward( this, X, ~, dZ, ~ )
            
            X = iWrapSingleInputInCell(X);
            
            % Calculate the size along the concatenation axis for each
            % input
            inputSizeAlongConcatenationAxis = ...
                iGetSizeForEachInputForSpecifiedDimension( ...
                X, ...
                this.ConcatenationAxis);
            
            % Distribute the gradients.
            dX = iDistributeGradient( ...
                dZ, ...
                this.ConcatenationAxis, ...
                inputSizeAlongConcatenationAxis);
            
            dX = iUnwrapSingleInputInCell(dX);
            
            % There are no learnable parameters.
            dW = [];
        end
        
        function outputSize = forwardPropagateSize(this, inputSize)
            inputSize = iWrapSingleInputInCell(inputSize);
            
            % For concatenation, there is an edge case we must deal with
            % where a user wants to concatenate something that is [H W]
            % with something that is [H W C]. To address this correctly, we
            % pad all input sizes with ones, so that [H W] will then become
            % [H W 1].
            inputSize = iPadInputSizes(inputSize);
            
            outputSize = inputSize{1};
            sumOfSizeAlongConcatenationAxis = sum(cellfun(@(x)x(this.ConcatenationAxis), inputSize));
            outputSize(this.ConcatenationAxis) = sumOfSizeAlongConcatenationAxis;
        end
        
        function this = inferSize(this, ~)
        end
        
        function tf = isValidInputSize(this, inputSize)
            
            inputSize = iWrapSingleInputInCell(inputSize);
            
            % For concatenation, there is an edge case we must deal with
            % where a user wants to concatenate something that is [H W]
            % with something that is [H W C]. To address this correctly, we
            % pad all input sizes with ones, so that [H W] will then become
            % [H W 1].
            inputSize = iPadInputSizes(inputSize);
            
            % The ordering of the operations here matter. We use short
            % circuiting to place more general checks first.
            tf = iNumInputsMatchLayersInputs(inputSize, this.NumInputs) && ...
                iConcatenationAxisDoesNotExceedNumberOfDimensions(this.ConcatenationAxis, inputSize) && ...
                iInputsHaveSameSizeAlongOtherAxes(this.ConcatenationAxis, inputSize);
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
        
        function val = get.InputNames(this)
            val = iGenerateInputNames(this.NumInputs);
        end
    end
end

function X = iWrapSingleInputInCell(X)
% A concatenation layer could technically only take one input. In these
% situations, the input will not be in a cell array, so we must wrap it in
% a cell array.
if ~iscell(X)
    X = {X};
end
end

function tf = iNumInputsMatchLayersInputs(inputSize, layersNumInputs)
numInputs = numel(inputSize);
tf = numInputs == layersNumInputs;
end

function inputSizeAlongSpecifiedDimension = iGetSizeForEachInputForSpecifiedDimension( ...
    X, specifiedDimension)
numInputs = numel(X);
inputSizeAlongSpecifiedDimension = zeros(1,numInputs);
for i = 1:numInputs
    inputSizeAlongSpecifiedDimension(i) = size(X{i}, specifiedDimension);
end
end

function dX = iDistributeGradient(dZ, concatenationAxis, inputSizeAlongConcatenationAxis)

% Deal out the gradients to each input
divisionDimensions = num2cell(size(dZ));
divisionDimensions{concatenationAxis} = inputSizeAlongConcatenationAxis;
dX = mat2cell(dZ, divisionDimensions{:});

% The resulting cell array will have leading singleton dimensions. Reshape
% to remove these.
numInputs = numel(inputSizeAlongConcatenationAxis);
dX = reshape(dX, [1 numInputs]);
end

function paddedInputSize = iPadInputSizes(inputSize)
maxLength = max(cellfun(@length, inputSize));
lengths = cellfun(@length, inputSize, 'UniformOutput', false);
lengthsToPad = cellfun(@(x)maxLength-x, lengths, 'UniformOutput', false);
paddedInputSize = cellfun(@(x,y)[x ones([1 y])], inputSize, lengthsToPad, 'UniformOutput', false);
end

function tf = iConcatenationAxisDoesNotExceedNumberOfDimensions(concatenationAxis, inputSize)
numDimensions = length(inputSize{1});
tf = concatenationAxis <= numDimensions;
end

function tf = iInputsHaveSameSizeAlongOtherAxes(concatenationAxis, inputSize)
% Check that input sizes are the same along all the dimensions that are not
% the concatenation axis. Note that this function assumes that all the
% input sizes have the same length (i.e. the number of dimensions are the
% same for all inputs).
numInputs = numel(inputSize);
numDimensions = length(inputSize{1});
sizesAreEqual = true([1 numDimensions]);
for i = 2:numInputs
    theseSizesAreEqual = inputSize{i-1} == inputSize{i};
    sizesAreEqual = sizesAreEqual & theseSizesAreEqual;
end
sizesAreEqual(concatenationAxis) = [];
tf = all(sizesAreEqual);
end

function X = iUnwrapSingleInputInCell(X)
if iscell(X) && (numel(X) == 1)
    X = X{1};
end
end

function inputNames = iGenerateInputNames(numInputs)
inputNames = cellstr("in" + (1:numInputs));
end