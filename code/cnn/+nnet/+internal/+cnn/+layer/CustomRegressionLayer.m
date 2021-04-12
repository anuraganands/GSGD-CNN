classdef CustomRegressionLayer < nnet.internal.cnn.layer.RegressionLayer
    % CustomRegressionLayer    Internal custom regression layer
    
    %   Copyright 2017 The MathWorks, Inc.
    
    properties
        % LearnableParameters   Learnable parameters for the layer
        %   Output layers have no learnable parameters
        LearnableParameters = nnet.internal.cnn.layer.learnable.PredictionLearnableParameter.empty();
    end
    
    properties (Dependent)
        % Name (char array)   A name for the layer
        Name
        
        % ResponseNames (cellstr)   The names of the responses
        ResponseNames
    end
    
    properties (Constant)
        % DefaultName   Default layer's name. This will be assigned in case
        % the user leaves an empty name.
        DefaultName = 'regressionoutput'
    end
    
    properties (SetAccess = private)
        % HasSizeDetermined   Always true for custom layers
        HasSizeDetermined = true;
        
        % ExternalCustomLayer (nnet.layer.RegressionLayer)   The
        % corresponding external custom layer
        ExternalCustomLayer
        
        % LayerVerifier
        % (nnet.internal.cnn.layer.util.CustomLayerVerifier)
        LayerVerifier
    end
    
    methods
        function this = CustomRegressionLayer( anExternalCustomLayer, aLayerVerifier )
            this.ExternalCustomLayer = anExternalCustomLayer;
            this.LayerVerifier = aLayerVerifier;
        end
        
        function outputSize = forwardPropagateSize(~, inputSize)
            % forwardPropagateSize  This layer should not modify the input
            % size of the data when forwarding data through
            outputSize = inputSize;
        end
        
        function this = inferSize(this, ~)
            
            % no-op since this layer has nothing that can be inferred
        end
        
        function tf = isValidInputSize(~, ~)
            % We cannot do any check in advance on the input size of a
            % custom output layer
            tf = true;
        end
        
        function this = initializeLearnableParameters(this, ~)
            % initializeLearnableParameters     no-op since there are no
            % learnable parameters
        end
        
        function this = prepareForTraining(this)
            this.LearnableParameters = nnet.internal.cnn.layer.learnable.TrainingLearnableParameter.empty();
        end
        
        function this = prepareForPrediction(this)
            this.LearnableParameters = nnet.internal.cnn.layer.learnable.PredictionLearnableParameter.empty();
        end
        
        function this = setupForHostPrediction(this)
            % setupForHostPrediction     no-op since there are no learnable
            % parameters
        end
        
        function this = setupForGPUPrediction(this)
            % setupForGPUPrediction     no-op since there are no learnable
            % parameters
        end
        
        function this = setupForHostTraining(this)
            % setupForHostTraining     no-op since there are no learnable
            % parameters
        end
        
        function this = setupForGPUTraining(this)
            % setupForGPUTraining     no-op since there are no learnable
            % parameters
        end
        
        function loss = forwardLoss( this, Y, T )
            % forwardLoss    Return the loss between the output obtained from
            % the network and the expected output
            try
                loss = forwardLoss( this.ExternalCustomLayer, Y, T );
            catch cause
                iThrowWithCause(cause, 'nnet_cnn:internal:cnn:layer:CustomRegressionLayer:ForwardLossErrored', class(this.ExternalCustomLayer))
            end
            this.LayerVerifier.verifyForwardLossSize( loss );
            this.LayerVerifier.verifyForwardLossType( Y, loss );
        end
        
        function dX = backwardLoss( this, Y, T )
            % backwardLoss    Back propagate the derivative of the loss
            % function
            try
                dX = backwardLoss( this.ExternalCustomLayer, Y, T );
            catch cause
                iThrowWithCause(cause, 'nnet_cnn:internal:cnn:layer:CustomRegressionLayer:BackwardLossErrored', class(this.ExternalCustomLayer))
            end
            this.LayerVerifier.verifyBackwardLossSize( Y, dX );
            this.LayerVerifier.verifyBackwardLossType( Y, dX );
        end
        
        function this = set.Name( this, val )
            this.ExternalCustomLayer.Name = val;
        end
        
        function val = get.Name( this )
            val = this.ExternalCustomLayer.Name;
        end
        
        function this = set.ResponseNames( this, val )
            this.ExternalCustomLayer.ResponseNames = val;
        end
        
        function val = get.ResponseNames( this )
            val = this.ExternalCustomLayer.ResponseNames;
        end
    end
end

function iThrowWithCause( cause, errorID, varargin )
exception = MException( message( errorID, varargin{:} ) );
exception = exception.addCause( cause );
throwAsCaller( exception );
end