classdef BiLSTMLayer < nnet.cnn.layer.Layer & nnet.internal.cnn.layer.Externalizable
    % BiLSTMLayer  Bidirectional Long Short-Term Memory (BiLSTM) layer
    %
    % To create a BiLSTM layer, use bilstmLayer.
    %
    % BiLSTMLayer properties:
    %     Name                                - Name of the layer
    %     InputSize                           - Input size of the layer
    %     NumHiddenUnits                      - Number of hidden units in
    %                                           the forward and backward
    %                                           LSTMs
    %     OutputMode                          - Output mode of the layer
    %
    % Properties for learnable parameters:
    %     InputWeights                        - Input weights
    %     InputWeightsLearnRateFactor         - Learning rate multiplier
    %                                           for the input weights
    %     InputWeightsL2Factor                - L2 multiplier for the
    %                                           input weights
    %
    %     RecurrentWeights                    - Recurrent weights
    %     RecurrentWeightsLearnRateFactor     - Learning rate multiplier
    %                                           for the recurrent weights
    %     RecurrentWeightsL2Factor            - L2 multiplier for the
    %                                           recurrent weights
    %
    %     Bias                                - Bias vector
    %     BiasLearnRateFactor                 - Learning rate multiplier
    %                                           for the bias
    %     BiasL2Factor                        - L2 multiplier for the bias
    %                                       
    % State parameters:
    %     HiddenState                         - Hidden state vector
    %     CellState                           - Cell state vector
    %
    %   Example:
    %       Create a Bidirectional Long Short-Term Memory layer.
    %
    %       layer = bilstmLayer(10)
    %
    %   See also bilstmLayer
    
    %   Copyright 2017 The MathWorks, Inc.
    
    properties(Dependent)
        % Name   A name for the layer
        %   The name for the layer. If this is set to '', then a name will
        %   be automatically set at training time.
        Name         
    end
    
    properties(SetAccess = private, Dependent)
        % InputSize   The input size for the layer. If this is set to
        % 'auto', then the input size will be automatically set during
        % training
        InputSize
        
        % NumHiddenUnits   The number of hidden units in the forward
        % sequence and backward sequence LSTMs
        NumHiddenUnits
        
        % OutputMode   The output format of the layer. If 'sequence',
        % output is a sequence. If 'last', the output is the last element
        % in a sequence
        OutputMode
    end
    
    properties(Dependent)
        % InputWeights   Input weights for the layer
        %   The input weight matrix is a vertical concatenation of the
        %   eight "gate" input weight matrices in the bidirectional LSTM.
        %   The forward LSTM gates are concatenated on top of the backward
        %   LSTM gates. Those individual matrices are concatenated in the
        %   following order: input gate, forget gate, layer input, output
        %   gate. This matrix will have size 8*NumHiddenUnits-by-InputSize.
        InputWeights
        
        % InputWeightsLearnRateFactor   Learning rate factor for the input
        % weights
        %   The learning rate factor is multiplied with the global learning
        %   rate to determine the learning rate for the input weights in
        %   this layer. For example, if it is set to 2, then the learning
        %   rate for the input weights in this layer will be twice the
        %   current global learning rate. To control the value of the learn
        %   rate for the eight individual matrices in the InputWeights,
        %   assign a 1-by-8 vector.
        InputWeightsLearnRateFactor (1,:) {mustBeNumeric, iCheckFactorDimensions}
        
        % InputWeightsL2Factor   L2 regularization factor for the input
        % weights
        %   The L2 regularization factor is multiplied with the global L2
        %   regularization setting to determine the L2 regularization
        %   setting for the input weights in this layer. For example, if it
        %   is set to 2, then the L2 regularization for the input weights
        %   in this layer will be twice the global L2 regularization
        %   setting. To control the value of the L2 factor for the eight
        %   individual matrices in the InputWeights, assign a 1-by-8
        %   vector.
        InputWeightsL2Factor (1,:) {mustBeNumeric, iCheckFactorDimensions}
        
        % RecurrentWeights   Recurrent weights for the layer
        %   The recurrent weight matrix is a vertical concatenation of the
        %   eight "gate" recurrent weight matrices in the bidirectional
        %   LSTM. The forward LSTM gates are concatenated on top of the
        %   backward LSTM gates. Those individual matrices are concatenated
        %   in the following order: input gate, forget gate, layer input,
        %   output gate. This matrix will have size
        %   8*NumHiddenUnits-by-NumHiddenUnits.
        RecurrentWeights
        
        % RecurrentWeightsLearnRateFactor   Learning rate factor for the
        % recurrent weights
        %   The learning rate factor is multiplied with the global learning
        %   rate to determine the learning rate for the recurrent weights
        %   in this layer. For example, if it is set to 2, then the
        %   learning rate for the recurrent weights in this layer will be
        %   twice the current global learning rate. To control the value of
        %   the learn rate for the eight individual matrices in the
        %   RecurrentWeights, assign a 1-by-8 vector.
        RecurrentWeightsLearnRateFactor (1,:) {mustBeNumeric, iCheckFactorDimensions}
        
        % RecurrentWeightsL2Factor   L2 regularization factor for the
        % recurrent weights
        %   The L2 regularization factor is multiplied with the global L2
        %   regularization setting to determine the L2 regularization
        %   setting for the recurrent weights in this layer. For example,
        %   if it is set to 2, then the L2 regularization for the recurrent
        %   weights in this layer will be twice the global L2
        %   regularization setting. To control the value of the L2 factor
        %   for the eight individual matrices in the RecurrentWeights,
        %   assign a 1-by-8 vector.
        RecurrentWeightsL2Factor (1,:) {mustBeNumeric, iCheckFactorDimensions}
        
        % Bias   Biases for the layer
        %   The bias vector is a concatenation of the eight "gate" bias
        %   vectors in the bidirectional LSTM. The forward LSTM gates are
        %   concatenated on top of the backward LSTM gates.  Those
        %   individual vectors are concatenated in the following order:
        %   input gate, forget gate, layer input, output gate. This vector
        %   will have size 8*NumHiddenUnits-by-1.
        Bias
        
        % BiasLearnRateFactor   Learning rate factor for the biases
        %   The learning rate factor is multiplied with the global learning
        %   rate to determine the learning rate for the bias in this layer.
        %   For example, if it is set to 2, then the learning rate for the
        %   bias in this layer will be twice the current global learning
        %   rate. To control the value of the learn rate for the eight
        %   individual vectors in the Bias, assign a 1-by-8 vector.
        BiasLearnRateFactor (1,:) {mustBeNumeric, iCheckFactorDimensions}
        
        % BiasL2Factor   L2 regularization factor for the biases
        %   The L2 regularization factor is multiplied with the global L2
        %   regularization setting to determine the L2 regularization
        %   setting for the biases in this layer. For example, if it is set
        %   to 2, then the L2 regularization for the biases in this layer
        %   will be twice the global L2 regularization setting. To control
        %   the value of the L2 factor for the eight individual vectors in
        %   the Bias, assign a 1-by-8 vector.
        BiasL2Factor (1,:) {mustBeNumeric, iCheckFactorDimensions}
    end
    
    properties(SetAccess = private, Dependent)
        % HiddenState   The initial value of the output state.
        %   The initial value of the output state. This vector will have
        %   size 2*NumHiddenUnits-by-1.
        HiddenState
        
        % CellState   The initial value of the cell state
        %   The initial value of the cell state. This vector will have size
        %   2*NumHiddenUnits-by-1.
        CellState
    end
    
    methods
        function this = BiLSTMLayer(privateLayer)
            this.PrivateLayer = privateLayer;
        end
        
        function val = get.Name(this)
            val = this.PrivateLayer.Name;
        end
        
        function this = set.Name(this, val)
            iAssertValidLayerName(val);
            this.PrivateLayer.Name = char(val);
        end
        
        function val = get.InputSize(this)
            val = this.PrivateLayer.InputSize;
            if isempty(val)
                val = 'auto';
            end
        end
        
        function val = get.NumHiddenUnits(this)
            val = this.PrivateLayer.HiddenSize;
        end
        
        function val = get.OutputMode(this)
            val = iGetOutputMode( this.PrivateLayer.ReturnSequence );
        end
        
        function val = get.InputWeights(this)
            val = this.PrivateLayer.InputWeights.HostValue;
        end
        
        function this = set.InputWeights(this, value)
            classes = {'single', 'double', 'gpuArray'};
            if isequal(this.InputSize, 'auto')
                expectedInputSize = NaN;
            else
                expectedInputSize = this.InputSize;
            end
            attributes = {'size', [8*this.NumHiddenUnits expectedInputSize], 'nonempty', 'real', 'nonsparse'};
            validateattributes(value, classes, attributes);
            
            this.PrivateLayer = this.PrivateLayer.inferSize( size(value, 2) );
            this.PrivateLayer.InputWeights.Value = gather(value);
        end
        
        function val = get.RecurrentWeights(this)
            val = this.PrivateLayer.RecurrentWeights.HostValue;
        end
        
        function this = set.RecurrentWeights(this, value)
            classes = {'single', 'double', 'gpuArray'};
            attributes = {'size', [8*this.NumHiddenUnits this.NumHiddenUnits], 'nonempty', 'real', 'nonsparse'};
            validateattributes(value, classes, attributes);
            
            this.PrivateLayer.RecurrentWeights.Value = gather(value);
        end
        
        function val = get.Bias(this)
            val = this.PrivateLayer.Bias.HostValue;
        end
        
        function this = set.Bias(this, value)
            classes = {'single', 'double', 'gpuArray'};
            attributes = {'column', 'nonempty', 'real', 'nonsparse', 'nrows', 8*this.NumHiddenUnits};
            validateattributes(value, classes, attributes);
            
            this.PrivateLayer.Bias.Value = gather(value);
        end
       
        function val = get.CellState(this)
            val = gather(this.PrivateLayer.CellState.Value);
        end
        
        function val = get.HiddenState(this)
            val = gather(this.PrivateLayer.HiddenState.Value);
        end
          
        function val = get.InputWeightsLearnRateFactor(this)
            val = this.getFactor(this.PrivateLayer.InputWeights.LearnRateFactor);
        end
        function this = set.InputWeightsLearnRateFactor(this, val)
            iAssertValidFactor(val)
            this.PrivateLayer.InputWeights.LearnRateFactor = this.setFactor(val);
        end
        
        function val = get.InputWeightsL2Factor(this)
            val = this.getFactor(this.PrivateLayer.InputWeights.L2Factor);
        end
        function this = set.InputWeightsL2Factor(this, val)
            iAssertValidFactor(val)
            this.PrivateLayer.InputWeights.L2Factor = this.setFactor(val);
        end
        
        function val = get.RecurrentWeightsLearnRateFactor(this)
            val = this.getFactor(this.PrivateLayer.RecurrentWeights.LearnRateFactor);
        end
        function this = set.RecurrentWeightsLearnRateFactor(this, val)
            iAssertValidFactor(val)
            this.PrivateLayer.RecurrentWeights.LearnRateFactor = this.setFactor(val);
        end
        
        function val = get.RecurrentWeightsL2Factor(this)
            val = this.getFactor(this.PrivateLayer.RecurrentWeights.L2Factor);
        end
        function this = set.RecurrentWeightsL2Factor(this, val)
            iAssertValidFactor(val)
            this.PrivateLayer.RecurrentWeights.L2Factor = this.setFactor(val);
        end
        
        function val = get.BiasLearnRateFactor(this)
            val = this.getFactor(this.PrivateLayer.Bias.LearnRateFactor);
        end
        function this = set.BiasLearnRateFactor(this, val)
            iAssertValidFactor(val)
            this.PrivateLayer.Bias.LearnRateFactor = this.setFactor(val);
        end
        
        function val = get.BiasL2Factor(this)
            val = this.getFactor(this.PrivateLayer.Bias.L2Factor);
        end
        function this = set.BiasL2Factor(this, val)
            iAssertValidFactor(val)
            this.PrivateLayer.Bias.L2Factor = this.setFactor(val);
        end
        
        function out = saveobj(this)
            privateLayer = this.PrivateLayer;
            out.Version = 1.0;
            out.Name = privateLayer.Name;
            out.InputSize = privateLayer.InputSize;
            out.NumHiddenUnits = privateLayer.HiddenSize;
            out.ReturnSequence = privateLayer.ReturnSequence;
            out.InputWeights = toStruct(privateLayer.InputWeights);
            out.RecurrentWeights = toStruct(privateLayer.RecurrentWeights);
            out.Bias = toStruct(privateLayer.Bias);
            out.CellState = toStruct(privateLayer.CellState);
            out.HiddenState = toStruct(privateLayer.HiddenState);
            out.InitialCellState = gather(privateLayer.InitialCellState);
            out.InitialHiddenState = gather(privateLayer.InitialHiddenState);
        end
    end
   
    methods(Static)
        function inputArguments = parseInputArguments(varargin)
            parser = iCreateParser();
            parser.parse(varargin{:});
            inputArguments = parser.Results;
            inputArguments.InputSize = [];
        end
        
        function this = loadobj(in)
            internalLayer = nnet.internal.cnn.layer.BiLSTM( in.Name, ...
                in.InputSize, ...
                in.NumHiddenUnits, ...
                true, ...
                true, ...
                in.ReturnSequence );
            internalLayer.InputWeights = nnet.internal.cnn.layer.learnable.PredictionLearnableParameter.fromStruct(in.InputWeights);
            internalLayer.RecurrentWeights = nnet.internal.cnn.layer.learnable.PredictionLearnableParameter.fromStruct(in.RecurrentWeights);
            internalLayer.Bias = nnet.internal.cnn.layer.learnable.PredictionLearnableParameter.fromStruct(in.Bias);
            internalLayer.CellState = nnet.internal.cnn.layer.dynamic.TrainingDynamicParameter.fromStruct(in.CellState);
            internalLayer.HiddenState = nnet.internal.cnn.layer.dynamic.TrainingDynamicParameter.fromStruct(in.HiddenState);
            internalLayer.InitialHiddenState = in.InitialHiddenState;
            internalLayer.InitialCellState = in.InitialCellState;

            this = nnet.cnn.layer.BiLSTMLayer(internalLayer);
        end
    end
    
    methods(Hidden, Access = protected)
        function [description, type] = getOneLineDisplay(obj)
            description = iGetMessageString( ...
                'nnet_cnn:layer:BiLSTMLayer:oneLineDisplay', ...
                num2str(obj.NumHiddenUnits));
            
            type = iGetMessageString( 'nnet_cnn:layer:BiLSTMLayer:Type' );
        end
        
        function groups = getPropertyGroups( this )
            generalParameters = { 'Name' };
            hyperParameters = { 'InputSize', ...
                'NumHiddenUnits', ...
                'OutputMode' };
            learnableParameters = { 'InputWeights', ...
                'RecurrentWeights', ...
                'Bias' };
            stateParameters = { 'HiddenState', 'CellState' };
            groups = [
                this.propertyGroupGeneral( generalParameters )
                this.propertyGroupHyperparameters( hyperParameters )
                this.propertyGroupLearnableParameters( learnableParameters )
                this.propertyGroupDynamicParameters( stateParameters )
                ];
        end
                
        function footer = getFooter( this )
            variableName = inputname(1);
            footer = this.createShowAllPropertiesFooter( variableName );
        end
        
        function val = getFactor(this, val)
            if isscalar(val)
                % No operation needed
            elseif numel(val) == (8*this.NumHiddenUnits)
                val = val(1:this.NumHiddenUnits:end);
                val = val(:)';
            else
                % Error - the factor has incorrect size
            end
        end
        
        function val = setFactor(this, val)
            if isscalar(val)
                % No operation needed
            elseif numel(val) == 8
                % Expand an eight-element vector into a 8*NumHiddenUnits-by-1
                % column vector
                expandedValues = repelem( val, this.NumHiddenUnits );
                val = expandedValues(:);
            else
                % Error - the factor has incorrect size
            end
        end
    end
end

function messageString = iGetMessageString( varargin )
messageString = getString( message( varargin{:} ) );
end

function p = iCreateParser()
p = inputParser;

defaultName = '';
defaultOutputMode = 'sequence';
defaultWeightLearnRateFactor = 1;
defaultBiasLearnRateFactor = 1;
defaultWeightL2Factor = 1;
defaultBiasL2Factor = 0;

p.addRequired('NumHiddenUnits', @(x)validateattributes(x, {'numeric'}, {'scalar', 'positive', 'integer'}));
p.addParameter('Name', defaultName, @nnet.internal.cnn.layer.paramvalidation.validateLayerName);
p.addParameter('OutputMode', defaultOutputMode, @(x)iAssertValidOutputMode(x), 'PartialMatchPriority', 1);
p.addParameter('InputWeightsLearnRateFactor', defaultWeightLearnRateFactor, @(x)iAssertValidFactor(x));
p.addParameter('RecurrentWeightsLearnRateFactor', defaultWeightLearnRateFactor,@(x)iAssertValidFactor(x));
p.addParameter('BiasLearnRateFactor', defaultBiasLearnRateFactor,@(x)iAssertValidFactor(x));
p.addParameter('InputWeightsL2Factor', defaultWeightL2Factor, @(x)iAssertValidFactor(x));
p.addParameter('RecurrentWeightsL2Factor', defaultWeightL2Factor, @(x)iAssertValidFactor(x));
p.addParameter('BiasL2Factor', defaultBiasL2Factor, @(x)iAssertValidFactor(x));
end

function mode = iGetOutputMode( tf )
if tf
    mode = 'sequence';
else
    mode = 'last';
end
end

function iCheckFactorDimensions( value )
dim = numel( value );
if ~(dim == 1 || dim == 8)
    exception = MException(message('nnet_cnn:layer:BiLSTMLayer:InvalidFactor'));
    throwAsCaller(exception);
end
end

function iAssertValidOutputMode(value)
validatestring(value, {'sequence', 'last'});
end

function iAssertValidFactor(value)
validateattributes(value, {'numeric'},  {'vector', 'real', 'nonnegative', 'finite'});
end

function iAssertValidLayerName(name)
nnet.internal.cnn.layer.paramvalidation.validateLayerName(name);
end