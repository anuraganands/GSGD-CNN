function layer = leakyReluLayer(varargin)
% leakyReluLayer   Leaky rectified linear unit (ReLU) layer
%
%   layer = leakyReluLayer(scale) creates a leaky rectified linear unit
%   layer. This type of layer performs a simple threshold operation,
%   where any input value less than zero is multiplied by a scalar
%   multiple. This is equivalent to:
%     out = in;        % For in>0
%     out = scale.*in; % For in<=0
%
%   layer = leakyReluLayer() uses a default scale of 0.01.
%
%   layer = leakyReluLayer(scale, 'PARAM1', VAL1) specifies optional
%   parameter name/value pairs for creating the layer:
%       'Name'  - A name for the layer. The default is ''.
%
%   Example:
%       Create a rectified linear unit layer.
%
%       layer = leakyReluLayer(0.1);
%
%   See also reluLayer, clippedReluLayer, nnet.cnn.layer.LeakyReLULayer.

%   Copyright 2016-2017 The MathWorks, Inc.

% Parse the input arguments.
inputArguments = iParseInputArguments(varargin{:});

% Create an internal representation of a ReLU layer.
internalLayer = nnet.internal.cnn.layer.LeakyReLU(inputArguments.Name, inputArguments.Scale);

% Pass the internal layer to a function to construct a user visible LeakyReLU
% layer.
layer = nnet.cnn.layer.LeakyReLULayer(internalLayer);

end

function inputArguments = iParseInputArguments(varargin)
parser = iCreateParser();
parser.parse(varargin{:});
inputArguments = parser.Results;
end

function p = iCreateParser()
p = inputParser;

defaultName = '';
defaultScale = 0.01;

addOptional(p, 'Scale', defaultScale, @iAssertValidScale);
addParameter(p, 'Name', defaultName, @iAssertValidLayerName);
end

function iAssertValidScale(value)
validateattributes(value, {'numeric'}, ...
    {'scalar','finite'});
end

function iAssertValidLayerName(name)
nnet.internal.cnn.layer.paramvalidation.validateLayerName(name)
end
