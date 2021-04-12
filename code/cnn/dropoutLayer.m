function layer = dropoutLayer( varargin )
% dropoutLayer   Dropout layer
%
%   layer = dropoutLayer() creates a dropout layer. During training, the
%   dropout layer will randomly set input elements to zero with a
%   probability of 0.5. This can be useful to prevent overfitting.
%
%   layer = dropoutLayer(probability) will create a dropout layer, where
%   probability is a number between 0 and 1 which specifies the probability
%   that an element will be set to zero. The default is 0.5.
%
%   layer = dropoutLayer(probability, 'PARAM1', VAL1) specifies optional
%   parameter name/value pairs for creating the layer:
%
%       'Name'                    - A name for the layer. The default is
%                                   ''.
%
%   It is important to note that when creating a network, dropout will only
%   be used during training.
%
%   Example:
%       Create a dropout layer which will dropout roughly 40% of the input
%       elements.
%
%       layer = dropoutLayer(0.4);
%
%   See also nnet.cnn.layer.DropoutLayer, imageInputLayer, reluLayer.

%   Copyright 2015-2017 The MathWorks, Inc.

% Parse the input arguments.
inputArguments = iParseInputArguments(varargin{:});

% Create an internal representation of a dropout layer.
internalLayer = nnet.internal.cnn.layer.Dropout( ...
    inputArguments.Name, ...
    inputArguments.Probability);

% Pass the internal layer to a  function to construct a user visible
% dropout layer.
layer = nnet.cnn.layer.DropoutLayer(internalLayer);

end

function inputArguments = iParseInputArguments(varargin)
parser = iCreateParser();
parser.parse(varargin{:});
inputArguments = iConvertToCanonicalForm(parser);
end

function p = iCreateParser()
p = inputParser;

defaultProbability = 0.5;
defaultName = '';

addOptional(p, 'Probability', defaultProbability, @iAssertValidProbability);
addParameter(p, 'Name', defaultName, @iAssertValidLayerName);
end

function iAssertValidLayerName(name)
nnet.internal.cnn.layer.paramvalidation.validateLayerName(name)
end

function iAssertValidProbability(value)
validateattributes(value, {'numeric'}, ...
    {'scalar','real','finite','>=',0,'<=',1});
end

function inputArguments = iConvertToCanonicalForm(p)
inputArguments = struct;
inputArguments.Probability = p.Results.Probability;
inputArguments.Name = char(p.Results.Name); % make sure strings get converted to char vectors
end
