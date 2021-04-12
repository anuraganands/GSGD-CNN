classdef (Abstract) Layer
    % Layer   Interface for convolutional neural network layers
    
    %   Copyright 2015-2017 The MathWorks, Inc.
    
    properties (Abstract)
        % LearnableParameters   Learnable parameters for the layer
        % (Vector of nnet.internal.cnn.layer.LearnableParameter)
        LearnableParameters
        
        % Name (char array)
        Name
    end
    
    properties (Abstract, SetAccess = private)
        % InputNames   The names of the inputs for this layer
        %   This should be defined as a row cell array of char arrays. If
        %   there is only one input, then it should still be in a cell.
        %   Note that the ordering of the names determines the order of the
        %   inputs.
        InputNames
        
        % OutputNames   The names of the outputs for this layer
        %   This should be defined as a row cell array of char arrays. If
        %   there is only one output, then it should still be in a cell.
        %   Note that the ordering of the names determines the order of the
        %   outputs.
        OutputNames
        
        % HasSizeDetermined   True for layers with size determined.
        HasSizeDetermined
    end
    
    properties (Abstract, Constant)
        % DefaultName   Default layer's name. This will be assigned in case
        % the user leaves an empty name.
        DefaultName
    end
    
    methods (Abstract)
        % predict   Forward input data through the layer and output the result
        %
        % Syntax
        %   Z = predict( aLayer, X )
        %
        % Inputs
        %   aLayer - the layer to forward through
        %   X - the input to forward propagate through the layer
        %
        % Output
        %   Z - the output of forward propagation through the layer
        Z = predict( aLayer, X )
        
        % backward    Back propagate the derivative of the loss function
        % through one layer
        %
        % For notation, assume that a layer is represented by the function
        % Z = f(X,W) where X is the input data to the layer, W the
        % learnable weights, and Z the layer output. The derivatives of the
        % loss function with respect to X, W, and Z are denoted dX, dW, and
        % dZ.
        %
        % Syntax
        %   [dX, dW] = backward( aLayer, X, Z, dZ, memory )
        %
        % Inputs
        %   aLayer - the layer to backprop through
        %   X - the input that was used for forward propagation through the
        %       layer
        %   Z - the output from forward propagation through the layer
        %   dZ - the derivative of the loss function with respect to Z.
        %       This is usually obtained via back-propagation from the next
        %       layer in the network.
        %   memory - whatever "memory" that was produced by forward
        %       propagation through the layer using the forward method
        %
        % Outputs
        %   dX  - the derivative of the loss function with respect to X
        %   dW  - cell array of derivatives of the loss function with
        %         respect to W with one element for each learnable
        %         parameter
        %
        % See also: forward
        varargout = backward( aLayer, X, Z, dZ, memory )
        
        % forwardPropagateSize    The size of the output from the layer for
        % a given size of input
        outputSize = forwardPropagateSize(this, inputSize)
        
        % inferSize    Infer the size of the learnable parameters based
        % on the input size
        this = inferSize(this, inputSize)
        
        % isValidInputSize   Check if the layer can accept an input of a
        % certain size
        tf = isValidInputSize(inputSize)
        
        % initializeLearnableParameters    Initialize learnable parameters
        % using their initializer
        this = initializeLearnableParameters(this, precision)
        
        % prepareForTraining   Prepare the layer for training
        this = prepareForTraining(this)
        
        % prepareForPrediction   Prepare the layer for prediction
        this = prepareForPrediction(this)
        
        % setupForHostPrediction   Prepare this layer for host prediction
        this = setupForHostPrediction(this)
        
        % setupForGPUPrediction   Prepare this layer for GPU prediction
        this = setupForGPUPrediction(this)
        
        % setupForHostTraining   Prepare this layer for host training
        this = setupForHostTraining(this)
        
        % setupForGPUTraining   Prepare this layer for GPU training
        this = setupForGPUTraining(this)
    end
    
    methods
        % forward   Forward input data through the layer and output the
        % result. If forward is not overridden, Z will be the output of
        % predict and memory will be an empty numeric array
        %
        % Syntax
        %   [Z, memory] = forward( aLayer, X )
        %
        % Inputs
        %   aLayer - the layer to forward through
        %   X - the input to forward propagate through the layer
        %
        % Outputs
        %   Z - the output of forward propagation through the layer
        %   memory - "memory" produced by forward propagation through the layer
        function [Z, memory] = forward( aLayer, X )
            Z = predict( aLayer, X );
            memory = [];
        end
        
        function this = moveToHost(this)
            % moveToHost  Move learnables from GPU to CPU
            %
            % Prepares the layer for use on the host by moving learnables
            % from the GPU.
            numParameters = numel(this.LearnableParameters);
            for i = 1:numParameters
                this.LearnableParameters(i).Value = gather(this.LearnableParameters(i).Value);
            end
        end
        function this = moveToGPU(this)
            % moveToCPU  Move learnables from CPU to GPU
            %
            % Prepares the layer for use on the GPU by moving learnables
            % to the GPU.
            numParameters = numel(this.LearnableParameters);
            for i = 1:numParameters
                this.LearnableParameters(i).Value = gpuArray(this.LearnableParameters(i).Value);
            end
        end
    end
    
    methods(Hidden, Static)
        function [layerIdx, layerNames] = findLayerByName(layers, name)
            % findLayerByName   Find index of layer by name
            %
            % Returns an index to layer that with a matching name. An empty is
            % returned if a layer with that name is not found. Multiple matches
            % may be returned. Callers must decide how to handle these cases.
            layerNames = nnet.internal.cnn.layer.Layer.getLayerNames(layers);
            layerIdx = find(strcmp(name, layerNames));
        end
        
        function layerNames = getLayerNames(layers)
            layerNames = cellfun(@(x)x.Name, layers, 'UniformOutput', false);
        end
    end
    
    methods
        % These are methods related to layer inputs and outputs. They are
        % placed here for convenience rather than putting them in a
        % package.
        function inputIndex = inputName2Index(this, inputName)
            inputIndex = find(strcmp(this.InputNames, inputName));
        end
        
        function outputIndex = outputName2Index(this, outputName)
            outputIndex = find(strcmp(this.OutputNames, outputName));
        end
        
        function inputName = inputIndex2Name(this, inputIndex)
            if isempty(this.InputNames)
                % We need to handle this case for the NetworkAnalyzer
                inputName = '';
            else
                inputName = this.InputNames{inputIndex};
            end
        end
        
        function outputName = outputIndex2Name(this, outputIndex)
            if isempty(this.OutputNames)
                % We need to handle this case for the NetworkAnalyzer
                outputName = '';
            else
                outputName = this.OutputNames{outputIndex};
            end
        end
        
        function val = numInputs(this)
            val = numel(this.InputNames);
        end
        
        function val = numOutputs(this)
            val = numel(this.OutputNames);
        end
    end
end
