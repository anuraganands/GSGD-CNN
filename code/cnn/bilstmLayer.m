function layer = bilstmLayer(varargin)
%bilstmLayer   Bidirectional Long Short-Term Memory (BiLSTM) layer
%
%   layer = bilstmLayer(numHiddenUnits) creates a Bidirectional Long
%   Short-Term Memory layer. numHiddenUnits is the number of hidden units
%   in the forward and backward sequence LSTMs of the layer, specified as a
%   positive integer.
%
%   layer = bilstmLayer(numHiddenUnits, 'PARAM1', VAL1, 'PARAM2', VAL2, ...)
%   specifies optional parameter name/value pairs for creating the layer:
%
%     'Name'                              - Name for the layer, specified
%                                           as a character vector. The
%                                           default value is ''.
%     'OutputMode'                        - The format of the output of the
%                                           layer. Options are:
%                                               - 'sequence', to output a
%                                               full sequence.
%                                               - 'last', to output the
%                                               last element only. 
%                                           The default value is
%                                           'sequence'.
%     'InputWeightsLearnRateFactor'       - Multiplier for the learning 
%                                           rate of the input weights,
%                                           specified as a scalar or a
%                                           1-by-8 row vector. The default
%                                           value is 1.
%     'RecurrentWeightsLearnRateFactor'   - Multiplier for the learning 
%                                           rate of the recurrent weights,
%                                           specified as a scalar or a
%                                           1-by-8 row vector. The default
%                                           value is 1.
%     'BiasLearnRateFactor'               - Multiplier for the learning 
%                                           rate of the bias, specified as
%                                           a scalar or a 1-by-8 row
%                                           vector. The default value is 1.
%     'InputWeightsL2Factor'              - Multiplier for the L2
%                                           regularizer of the input
%                                           weights, specified as a scalar
%                                           or a 1-by-8 row vector. The
%                                           default value is 1.
%     'RecurrentWeightsL2Factor'          - Multiplier for the L2
%                                           regularizer of the recurrent
%                                           weights, specified as a scalar
%                                           or a 1-by-8 row vector. The
%                                           default value is 1.
%     'BiasL2Factor'                      - Multiplier for the L2
%                                           regularizer of the bias,
%                                           specified as a scalar or a
%                                           1-by-8 row vector. The default
%                                           value is 0.
%
%   Example 1:
%       Create a Bidirectional LSTM layer with 100 hidden units.
%
%       layer = bilstmLayer(100);
%
%   Example 2:
%       Create a Bidirectional LSTM layer with 50 hidden units which
%       returns a single element. Manually initialize the recurrent weights
%       from a Gaussian distribution with standard deviation 0.01
%
%       numHiddenUnits = 50;
%       layer = bilstmLayer(numHiddenUnits, 'OutputMode', 'last');
%       layer.RecurrentWeights = randn([8*numHiddenUnits numHiddenUnits])*0.01;
%
%   See also nnet.cnn.layer.BiLSTMLayer

%   Copyright 2017 The MathWorks, Inc.

% Parse the input arguments.
args = nnet.cnn.layer.BiLSTMLayer.parseInputArguments(varargin{:});

% Create an internal representation of the layer.
internalLayer = nnet.internal.cnn.layer.BiLSTM(args.Name, ...
    args.InputSize, ...
    args.NumHiddenUnits, ...
    true, ...
    true, ...
    iGetReturnSequence(args.OutputMode));

% Use the internal layer to construct a user visible layer.
layer = nnet.cnn.layer.BiLSTMLayer(internalLayer);

% Set learn rate and L2 Factors.
layer.InputWeightsL2Factor = args.InputWeightsL2Factor;
layer.InputWeightsLearnRateFactor = args.InputWeightsLearnRateFactor;

layer.RecurrentWeightsL2Factor = args.RecurrentWeightsL2Factor;
layer.RecurrentWeightsLearnRateFactor = args.RecurrentWeightsLearnRateFactor;

layer.BiasL2Factor = args.BiasL2Factor;
layer.BiasLearnRateFactor = args.BiasLearnRateFactor;

end

function tf = iGetReturnSequence( mode )
tf = true;
if strcmp( mode, 'last' )
    tf = false;
end
end