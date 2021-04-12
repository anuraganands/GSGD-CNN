function layer = clippedReluLayer(varargin)
% clippedReluLayer   Clipped rectified linear unit (ReLU) layer
%
%   layer = clippedReluLayer(ceiling) creates a clipped rectified linear
%   unit layer. This type of layer performs a simple threshold operation,
%   where any input value less than zero will be set to zero and any value
%   above the clipping ceiling will be set to the clipping ceiling. This is
%   equivalent to:
%     out = 0;         % For in<0
%     out = in;        % For 0<=in<ceiling
%     out = ceiling;   % For in>=ceiling
%
%   layer = clippedReluLayer(ceiling, 'PARAM1', VAL1) specifies optional
%   parameter name/value pairs for creating the layer:
%       'Name'  - A name for the layer. The default is ''.
%
%   Example:
%       Create a rectified linear unit layer.
%
%       layer = clippedReluLayer(2);
%
%   See also reluLayer, leakyReluLayer, nnet.cnn.layer.ClippedReLULayer.

%   Copyright 2016-2017 The MathWorks, Inc.

% Parse the input arguments.
inputArguments = iParseInputArguments(varargin{:});

% Create an internal representation of a clipped ReLU layer.
internalLayer = nnet.internal.cnn.layer.ClippedReLU(inputArguments.Name, inputArguments.Ceiling);

% Pass the internal layer to a function to construct a user visible
% ClippedReLU layer.
layer = nnet.cnn.layer.ClippedReLULayer(internalLayer);

end

function inputArguments = iParseInputArguments(varargin)
parser = iCreateParser();
parser.parse(varargin{:});
inputArguments = parser.Results;
end

function p = iCreateParser()
p = inputParser;

defaultName = '';

addRequired(p, 'Ceiling', @iAssertValidCeiling);
addParameter(p, 'Name', defaultName, @iAssertValidLayerName);
end

function iAssertValidLayerName(name)
nnet.internal.cnn.layer.paramvalidation.validateLayerName(name)
end

function iAssertValidCeiling(value)
validateattributes(value, {'numeric'}, ...
    {'scalar','finite','positive'});
end