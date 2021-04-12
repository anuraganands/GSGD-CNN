function layer = sequenceInputLayer(varargin)
% sequenceInputLayer   Sequence input layer
%
%   layer = sequenceInputLayer(inputSize) defines a sequence input layer.
%   inputSize is the number of dimensions of the input sequence at each
%   time step. It must be a positive integer.
%
%   layer = sequenceInputLayer(inputSize, 'PARAM1', VAL1, 'PARAM2', VAL2, ...)
%   specifies optional parameter name/value pairs for creating the layer:
%
%    'Name'              A name for the layer.
%
%                        Default: ''
%
%   Example:
%       Create a sequence input layer for multi-dimensional time series
%       with 5 dimensions per time step.
%
%       layer = sequenceInputLayer(5);
%
%   See also nnet.cnn.layer.SequenceInputLayer

%   Copyright 2017 The MathWorks, Inc.

% Parse the input arguments.
inputArguments = iParseInputArguments(varargin{:});

% Create an internal representation of a sequence input layer.
internalLayer = nnet.internal.cnn.layer.SequenceInput( ...
    inputArguments.Name, ...
    inputArguments.InputSize );

% Pass the internal layer to a function to construct a user visible
% sequence input layer.
layer = nnet.cnn.layer.SequenceInputLayer(internalLayer);

end

function inputArguments = iParseInputArguments(varargin)
parser = iCreateParser();
parser.parse(varargin{:});
inputArguments = iConvertToCanonicalForm(parser.Results);
end

function p = iCreateParser(varargin)
p = inputParser;

defaultName = '';

addRequired(p, 'InputSize', @iAssertValidInputSize);
addParameter(p, 'Name', defaultName, @iAssertValidLayerName);
end

function iAssertValidInputSize(inputSize)
validateattributes(inputSize, {'numeric'}, {'scalar', 'positive', 'integer'})
end

function iAssertValidLayerName(name)
nnet.internal.cnn.layer.paramvalidation.validateLayerName(name)
end

function inputArguments = iConvertToCanonicalForm(params)
inputArguments = struct;
inputArguments.InputSize = params.InputSize;
inputArguments.Name = char(params.Name); % Make sure strings get converted to char vectors
end