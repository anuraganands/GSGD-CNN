classdef MeanSquaredError < nnet.internal.cnn.layer.RegressionLayer
    % MeanSquaredError   MeanSquaredError loss output layer
    
    %   Copyright 2016-2017 The MathWorks, Inc.
    
    properties
        % LearnableParameters   Learnable parameters for the layer
        %   This layer has no learnable parameters.
        LearnableParameters = nnet.internal.cnn.layer.learnable.PredictionLearnableParameter.empty();
        
        % Name (char array)   A name for the layer
        Name
        
        % ResponseNames (cellstr)   The names of the responses
        ResponseNames
    end
    
    properties (Constant)
        % DefaultName   Default layer's name.
        DefaultName = 'regressionoutput'
    end
    
    properties (SetAccess = private)
        % HasSizeDetermined   True for layers with size determined.
        HasSizeDetermined
        
        % ObservationDim (scalar int)   The dimension of the input data
        % along which holds the number of observations within the data.
        ObservationDim
    end
    
    methods
        function this = MeanSquaredError(name)
            % MeanSquaredError   Constructor for the layer
            this.Name = name;
            this.ResponseNames = {};
            this.HasSizeDetermined = false;
        end
        
        function outputSize = forwardPropagateSize(~, inputSize)
            % forwardPropagateSize  Output the size of the layer based on
            % the input size
            outputSize = inputSize;
        end
        
        function this = inferSize(this, inputSize)
            % Infer observation dimension from input size:
            %   - numel(inputSize) = 3 --> ObservationDim = 4
            %   - numel(inputSize) = 1 --> ObservationDim = 2
            this.ObservationDim = numel(inputSize) + 1;
            this.HasSizeDetermined = true;
        end
        
        function tf = isValidInputSize(this, inputSize)
            % isValidInputSize   Check if the layer can accept an input of
            % a certain size.
            numElems = numel(inputSize);
            if this.HasSizeDetermined && (this.ObservationDim == 4)
                tf = numElems == 3;
            elseif this.HasSizeDetermined && (this.ObservationDim == 2)
                tf = numElems == 1;
            else
                tf = numElems == 3 || numElems == 1;
            end
        end
        
        function this = initializeLearnableParameters(this, ~)
            
            % no-op since there are no learnable parameters
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
        
        function loss = forwardLoss( this, Y, T )
            % forwardLoss    Return the MSE loss between estimate
            % and true responses averaged by the number of observations
            %
            % Syntax:
            %   loss = layer.forwardLoss( Y, T );
            %
            % Inputs (image):
            %   Y   Predictions made by network, of size:
            %   height-by-width-by-numResponses-by-numObservations
            %   T   Targets (actual values), of size:
            %   height-by-width-by-numResponses-by-numObservations
            %
            % Inputs (sequence):
            %   Y   Predictions made by network, of size:
            %   numResponses-by-numObservations-by-seqLength
            %   T   Targets (actual values), of size:
            %   numResponses-by-numObservations-by-seqLength
            
            squares = 0.5*(Y-T).^2;
            sY = size( Y );
            numElements = prod( sY(this.ObservationDim:end) );
            loss = sum( squares (:) ) / numElements;
        end
        
        function dX = backwardLoss( this, Y, T )
            % backwardLoss    Back propagate the derivative of the loss
            % function
            %
            % Syntax:
            %   dX = layer.backwardLoss( Y, T );
            %
            % Inputs (image):
            %   Y   Predictions made by network, of size:
            %   height-by-width-by-numResponses-by-numObservations
            %   T   Targets (actual values), of size:
            %   height-by-width-by-numResponses-by-numObservations
            %
            % Inputs (sequence):
            %   Y   Predictions made by network, of size:
            %   numResponses-by-numObservations-by-seqLength
            %   T   Targets (actual values), of size:
            %   numResponses-by-numObservations-by-seqLength
            
            numObservations = size( Y, this.ObservationDim );
            dX = (Y - T)./numObservations;
        end
    end
    
    methods (Static)
        function layer = constructWithObservationDim( name, responseNames, observationDim )
            % constructWithObservationDim   Construct a mean squared error
            % layer with the observation dimension and response names
            % defined on construction
            layer = nnet.internal.cnn.layer.MeanSquaredError( name );
            layer.ResponseNames = responseNames;
            layer.ObservationDim = observationDim;
            if ~isempty( observationDim )
                layer.HasSizeDetermined = true;
            else
                layer.HasSizeDetermined = false;
            end
        end
    end
end
