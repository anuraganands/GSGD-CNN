classdef CustomLayer < nnet.internal.cnn.layer.Layer
    % CustomLayer    Internal custom convolutional neural network layer
    
    %   Copyright 2017 The MathWorks, Inc.
    
    properties
        % LearnableParameterNames   List of learnable parameter names for
        % the external layer
        LearnableParameterNames = {};
    end
    
    properties (Dependent)
        % Name (char array)
        Name
        
        % LearnableParameters   Learnable parameters for the layer
        % (Vector of nnet.internal.cnn.layer.LearnableParameter)
        LearnableParameters
    end
    
    properties (Constant)
        % DefaultName   Default layer's name. This will be assigned in case
        % the user leaves an empty name.
        DefaultName = 'layer';
    end
    
    properties(Access = private)
        % LearnableParameters   Private storage for LearnableParameters
        PrivateLearnableParameters = nnet.internal.cnn.layer.learnable.PredictionLearnableParameter.empty();
        
        % IsForwardDefined   True if 'forward' was overridden in the
        % external custom layer
        IsForwardDefined
    end
    
    properties (Access = private, Dependent)
        % NumParameters   Number of learnable parameters
        NumParameters
    end
    
    properties (SetAccess = private)
        % InputNames   This layer has a single input
        InputNames = {'in'}
        
        % OutputNames   This layer has a single output
        OutputNames = {'out'}
        
        % HasSizeDetermined   Always true for custom layers
        HasSizeDetermined = true;
        
        % ExternalCustomLayer (nnet.layer.Layer)   The
        % corresponding external custom layer
        ExternalCustomLayer
        
        % LayerVerifier
        % (nnet.internal.cnn.layer.util.CustomLayerVerifier)
        LayerVerifier
    end
    
    methods
        function this = CustomLayer( anExternalCustomLayer, aLayerVerifier )
            this.ExternalCustomLayer = anExternalCustomLayer;
            [parametersNames, parameters] = iDiscoverLearnableParameters( anExternalCustomLayer );
            this.LearnableParameterNames = parametersNames;
            this.LearnableParameters = parameters;
            aLayerVerifier.LayerClass = class( anExternalCustomLayer );
            aLayerVerifier.LearnableParametersNames = parametersNames;
            this.LayerVerifier = aLayerVerifier;
            this.IsForwardDefined = iIsForwardDefined( anExternalCustomLayer );
        end
        
        function Z = predict( this, X )
            % predict   Forward input data through the external custom layer
            % and output the result
            try
                Z = predict( this.ExternalCustomLayer, X );
            catch cause
                iThrowWithCause(cause, 'nnet_cnn:internal:cnn:layer:CustomLayer:PredictErrored', class(this.ExternalCustomLayer))
            end
            
            this.LayerVerifier.verifyPredictType( X, Z );
        end
        
        function [Z, memory] = forward( this, X )
            % forward   Forward input data through the external custom layer
            % and output the result
            try
                [Z, memory] = forward( this.ExternalCustomLayer, X );
            catch cause
                if this.IsForwardDefined
                    iThrowWithCause(cause, 'nnet_cnn:internal:cnn:layer:CustomLayer:ForwardErrored', class(this.ExternalCustomLayer))
                else
                    iThrowWithCause(cause, 'nnet_cnn:internal:cnn:layer:CustomLayer:PredictErrored', class(this.ExternalCustomLayer))
                end
            end
            
            if this.IsForwardDefined
                this.LayerVerifier.verifyForwardType( X, Z );
            else
                this.LayerVerifier.verifyPredictType( X, Z );
            end
        end
        
        function varargout = backward( this, X, Z, dZ, memory )
            % backward    Back propagate the derivative of the loss function
            % through the external custom layer
            try
                needsWeightGradients = nargout > 1;
                if ~needsWeightGradients
                    varargout{1} = backward( this.ExternalCustomLayer, X, Z, dZ, memory );
                else
                    [dX, dW{1:this.NumParameters}] = ...
                        backward( this.ExternalCustomLayer, X, Z, dZ, memory );
                    varargout = { dX, dW };
                end
            catch cause
                iThrowWithCause(cause, 'nnet_cnn:internal:cnn:layer:CustomLayer:BackwardErrored', class(this.ExternalCustomLayer))
            end
            
            this.LayerVerifier.verifyBackwardSize( X, this.LearnableParameters, varargout{:} );
            this.LayerVerifier.verifyBackwardType( X, varargout{:} );
        end
        
        function outputSize = forwardPropagateSize(this, inputSize)
            % forwardPropagateSize   The size of the output from the layer
            % for a given size of input
            
            % Initialize learnable parameters since 'forward' will use them
            precision = nnet.internal.cnn.util.Precision('single');
            this = this.initializeLearnableParameters(precision);
            
            % We then need to setup for host prediction. This is because if
            % the weights were already initialized, then they could be GPU
            % values
            this = this.prepareForPrediction();
            this = this.setupForHostPrediction();
            
            % If inputSize is scalar, we will not transform the output size
            % to 3D
            if isscalar( inputSize )
                fakeData = iSingleInputData( [inputSize 1] );
                output = this.forward( fakeData );
                % We need to remove the trailing singleton dimension, so we
                % can propagate a scalar forward
                outputSize = iRemoveTrailingSingletonDimension( size(output) );
            else
                fakeData = iSingleInputData( inputSize );
                output = this.forward( fakeData );
                % Size is always 3-D, even when last dimension is singleton
                outputSize = i3DSize(output);
            end
        end
        
        function this = inferSize(this, ~)
            % We cannot infer the size of a custom layer so this is a no-op
        end
        
        function tf = isValidInputSize(this, inputSize)
            % isValidInputSize   Validate inputSize by generating fake
            % data and trying to propagate it through the layer and back
            
            % Initialize learnable parameters since 'forward' will use them
            precision = nnet.internal.cnn.util.Precision('single');
            this = this.initializeLearnableParameters(precision);
            
            % We then need to setup for host prediction. This is because if
            % the weights were already initialized, then they could be GPU
            % values
            this = this.prepareForPrediction();
            this = this.setupForHostPrediction();
            
            tf = true;
            fakeData = iSingleInputData( inputSize );
            try
                [Z, memory] = this.forward( fakeData );
                dLdZ = iSingleInputData( size(Z) );
                % Must request the outputs or dLdW won't be computed
                [~, ~] = this.backward( fakeData, Z, dLdZ, memory );
            catch cause
                rethrow( cause );
            end
        end
        
        function this = initializeLearnableParameters(this, precision)
            % initializeLearnableParameters   Cast the learnable parameters
            % using the appropriate precision
            for ii=1:this.NumParameters
                % Cast the external learnable parmaeters
                this.ExternalCustomLayer.(this.LearnableParameterNames{ii}) = precision.cast( ...
                    this.ExternalCustomLayer.(this.LearnableParameterNames{ii}) );
                % Update the internal values
                this.LearnableParameters(ii).Value = this.ExternalCustomLayer.(this.LearnableParameterNames{ii});
            end
        end
        
        function this = prepareForTraining(this)
            % prepareForTraining   Prepare this layer for training
            %   Before this layer can be used for training, we need to
            %   convert the learnable parameters to use the class
            %   TrainingLearnableParameter.
            this.LearnableParameters = nnet.internal.cnn.layer.learnable.convert2training(this.LearnableParameters);
        end
        
        function this = prepareForPrediction(this)
            % prepareForPrediction   Prepare this layer for prediction
            %   Before this layer can be used for prediction, we need to
            %   convert the learnable parameters to use the class
            %   PredictionLearnableParameter.
            
            this.LearnableParameters = nnet.internal.cnn.layer.learnable.convert2prediction(this.LearnableParameters);
        end
        
        function this = setupForHostPrediction(this)
            for ii = 1:this.NumParameters
                this.LearnableParameters(ii).UseGPU = false;
            end
        end
        
        function this = setupForGPUPrediction(this)
            for ii = 1:this.NumParameters
                this.LearnableParameters(ii).UseGPU = true;
            end
        end
        
        function this = setupForHostTraining(this)
            % no-op
        end
        
        function this = setupForGPUTraining(this)
            % no-op
        end
        
        function name = get.Name(this)
            name = this.ExternalCustomLayer.Name;
        end
        
        function this = set.Name(this, aName)
            this.ExternalCustomLayer.Name = aName;
        end
        
        function learnableParameters = get.LearnableParameters(this)
            learnableParameters = this.PrivateLearnableParameters;
        end
        
        function this = set.LearnableParameters(this, learnableParameters)
            
            % Update internal learnable parameters
            this.PrivateLearnableParameters = learnableParameters;
            
            % Update external learnable parameters
            externalLayer = this.ExternalCustomLayer;
            names = this.LearnableParameterNames;
            for ii=1:this.NumParameters
                externalLayer.(names{ii}) = learnableParameters(ii).Value;
            end
            this.ExternalCustomLayer = externalLayer;
        end
        
        function numParameters = get.NumParameters( this )
            numParameters = numel( this.LearnableParameterNames );
        end
    end
end

function [names, parameters] = iDiscoverLearnableParameters( externalLayer )
% iDiscoverLearnableParameters   Discover learnable parameters
% declared by the user and add them to the internal learnable
% parameters and track them with a learnable parameters list

names = {};
parameters = nnet.internal.cnn.layer.learnable.PredictionLearnableParameter.empty();

propertyList = iPropertyList(externalLayer);
for ii=1:numel(propertyList)
    if isprop(propertyList(ii), 'Learnable') && propertyList(ii).Learnable
        name = propertyList(ii).Name;
        
        % Store the parameter name
        names{end+1} = name; %#ok<AGROW>
        
        % Create and store matching learnable parameter
        initialValue = externalLayer.(name);
        learnRateFactor = getLearnRateFactor( externalLayer, name );
        l2Factor = getL2Factor( externalLayer, name );
        parameter = iNewPredictionLearnableParameter( initialValue, learnRateFactor, l2Factor );
        parameters(end+1) = parameter; %#ok<AGROW>
    end
end
end

function learnableParameter = iNewPredictionLearnableParameter( value, learnRateFactor, l2Factor )
learnableParameter = nnet.internal.cnn.layer.learnable.PredictionLearnableParameter;
learnableParameter.Value = value;
learnableParameter.LearnRateFactor = learnRateFactor;
learnableParameter.L2Factor = l2Factor;
end


function propertyList = iPropertyList( layer )
metaLayer = metaclass(layer);
propertyList = metaLayer.PropertyList.findobj( ...
    'GetAccess', 'public', 'SetAccess', 'public');
end

function functionList = iFunctionList( layer )
metaLayer = metaclass(layer);
functionList = metaLayer.MethodList;
end

function tf = iIsForwardDefined( externalLayer )
% iIsForwardDefined   Return true if forward was overriden in the external
% custom layer
tf = false;
functionList = iFunctionList( externalLayer );
for ii=1:numel(functionList)
    currentFunction = functionList(ii);
    if isequal( currentFunction.Name, 'forward' )
        definingClass = currentFunction.DefiningClass.Name;
        % If the defining class of 'forward' was the Layer super-class,
        % then the function has not been overridden by the user
        tf = ~isequal( definingClass, 'nnet.layer.Layer' );
    end
end
end

function data = iSingleInputData( inputSize )
data = ones( inputSize, 'single' );
end

function sz = i3DSize(x)
sz = [size(x,1) size(x,2) size(x,3)];
end

function sz = iRemoveTrailingSingletonDimension( sz )
if sz(end) == 1
    sz = sz(1);
end
end

function iThrowWithCause( cause, errorID, varargin )
exception = MException( message( errorID, varargin{:} ) );
exception = exception.addCause( cause );
throwAsCaller( exception );
end