classdef CrossEntropy < nnet.internal.cnn.layer.ClassificationLayer
    % CrossEntropy   Cross entropy loss output layer
    
    %   Copyright 2015-2017 The MathWorks, Inc.
    
    properties
        % LearnableParameters   Learnable parameters for the layer
        %   This layer has no learnable parameters.
        LearnableParameters = nnet.internal.cnn.layer.learnable.PredictionLearnableParameter.empty();
        
        % Name (char array)   A name for the layer
        Name
        
        % Categories (column categorical array) The categories of the classes
        % It can store ordinality of the classes as well.
        Categories
    end
    
    properties (Constant)
        % DefaultName   Default layer's name.
        DefaultName = 'classoutput'
    end
    
    properties (SetAccess = private)
        % HasSizeDetermined   True for layers with size determined.
        HasSizeDetermined
        
        % NumClasses (scalar int)   Number of classes
        NumClasses
        
        % ObservationDim (scalar int)   The dimension of the input data
        % along which holds the number of observations within the data.
        ObservationDim
    end
    
    methods
        function this = CrossEntropy(name, numClasses)
            % Output  Constructor for the layer
            % creates an output layer with the following parameters:
            %
            %   name                - Name for the layer
            %   numClasses          - Number of classes. [] if it has to be
            %                       determined later
            this.Name = name;
            if(isempty(numClasses))
                this.NumClasses = [];
            else
                this.NumClasses = numClasses;
            end
            this.Categories = categorical();
            this.HasSizeDetermined = ~isempty( numClasses );
            this.ObservationDim = 4;
        end
        
        function outputSize = forwardPropagateSize(~, inputSize)
            % forwardPropagateSize  Output the size of the layer based on
            % the input size
            outputSize = inputSize;
        end
        
        function this = inferSize(this, inputSize)
            % inferSize    Infer the number of classes and the observation
            % dimension, based on the input dimensions
            this.NumClasses = inputSize(end);
            this.ObservationDim = numel(inputSize) + 1;
            this.HasSizeDetermined = true;
        end
        
        function tf = isValidInputSize(this, inputSize)
            % isValidInputSize   Check if the layer can accept an input of
            % a certain size.
            if isempty(this.NumClasses)
                tf = iIsValid3DInputSize(inputSize) || isscalar(inputSize);
            else
                tf = isequal(inputSize, [1 1 this.NumClasses]) ...
                    || isequal(inputSize, this.NumClasses);
            end
        end
        
        function this = initializeLearnableParameters(this, ~)
            % initializeLearnableParameters     no-op since there are no
            % learnable parameters
        end
        
        function this = set.Categories( this, val )
            % Set Categories as a column array.
            if isrow(val)
                val = val';
            end
            this.Categories = val;
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
            % forwardLoss    Return the cross entropy loss between estimate
            % and true responses averaged by the number of observations
            %
            % Syntax:
            %   loss = layer.forwardLoss( Y, T );
            %
            % Image Inputs:
            %   Y   Predictions made by network, 1-by-1-by-numClasses-by-numObs
            %   T   Targets (actual values), 1-by-1-by-numClasses-by-numObs
            %
            % Vector Inputs:
            %   Y   Predictions made by network, numClasses-by-numObs-by-seqLength
            %   T   Targets (actual values),  numClasses-by-numObs-by-seqLength
            
            sY = size( Y );
            numElems = prod( sY(this.ObservationDim:end) );
            loss = -sum( sum( sum(T.*log(nnet.internal.cnn.util.boundAwayFromZero(Y))) ./numElems ) );
        end
        
        function dX = backwardLoss( this, Y, T )
            % backwardLoss    Back propagate the derivative of the loss
            % function
            %
            % Syntax:
            %   dX = layer.backwardLoss( Y, T );
            %
            % Image Inputs:
            %   Y   Predictions made by network, 1-by-1-by-numClasses-by-numObs
            %   T   Targets (actual values), 1-by-1-by-numClasses-by-numObs
            %
            % Vector Inputs:
            %   Y   Predictions made by network,  numClasses-by-numObs-by-seqLength
            %   T   Targets (actual values),  numClasses-by-numObs-by-seqLength
            
            numObservations = size( Y, this.ObservationDim );
            dX = (-T./nnet.internal.cnn.util.boundAwayFromZero(Y))./numObservations;
        end
    end
    
    methods (Static)
        function layer = constructWithObservationDim( name, numClasses, categories, observationDim )
            % constructWithObservationDim   Construct a cross entropy layer
            % with the observation dimension and categories defined on
            % construction
            layer = nnet.internal.cnn.layer.CrossEntropy( name, numClasses );
            layer.Categories = categories;
            layer.ObservationDim = observationDim;
        end
    end
end

function tf = iIsValid3DInputSize(inputSize)
if numel(inputSize) == 3
    tf = isequal(inputSize(1:2), [1 1]);
else
    tf = false;
end
end
