classdef SequenceInput < nnet.internal.cnn.layer.InputLayer
    % SequenceInput   Sequence input layer
    
    %   Copyright 2017 The MathWorks, Inc.
    
    properties
        % LearnableParameters   Learnable parameters for the layer
        %   This layer has no learnable parameters.
        LearnableParameters = nnet.internal.cnn.layer.learnable.PredictionLearnableParameter.empty();
        
        % Name (char array)   A name for the layer
        Name
    end
    
    properties (Constant)
        % DefaultName   Default layer's name.
        DefaultName = 'sequenceinput'
    end
    
    properties (SetAccess = private)
        % HasSizeDetermined   True for layers with size determined.
        HasSizeDetermined
        
        % InputSize (int)     Size of the input [number of channels]
        InputSize
    end
   
    methods
        function this = SequenceInput(name, inputSize)
            % Input  Constructor for the layer
            this.Name = name;
            this.InputSize = inputSize;
            this.HasSizeDetermined = true;
        end
        
        function Z = predict( ~, X )
            % predict   Forward input data through the layer and output the result
            Z = X;
        end
        
        function [dX,dW] = backward( ~, ~, ~, ~, ~ )
            % backward  Return empty value
            dX = [];
            dW = [];
        end
        
        function outputSize = forwardPropagateSize(this, ~)
            % forwardPropagateSize  Output the size of the layer
            outputSize = this.InputSize;
        end
        
        function this = inferSize(this, ~)
            % inferSize     no-op since this layer has nothing that can be
            %               inferred
        end
        
        function tf = isValidInputSize(this, inputSize)
            % isValidInputSize   Check if the layer can accept an input of
            % a certain size
            tf = isequal( inputSize, this.InputSize );
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
    end
end

