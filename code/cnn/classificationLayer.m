function layer = classificationLayer( varargin )
% classificationLayer   Classification output layer for a neural network
%
%   layer = classificationLayer() creates a classification output layer for
%   a neural network. The classification output layer holds the name of the
%   loss function that is used for training the network, the size of the
%   output, and the class labels.
%
%   layer = classificationLayer('PARAM1', VAL1) specifies optional
%   parameter name/value pairs for creating the layer:
%
%       'Name'                    - A name for the layer. The default is
%                                   ''.
%
%   Example:
%       Create a classification output layer.
%
%       layer = classificationLayer();
%
%   See also nnet.cnn.layer.ClassificationOutputLayer, softmaxLayer.

%   Copyright 2015-2017 The MathWorks, Inc.

% Parse the input arguments
args = iParseInputArguments(varargin{:});

% Create an internal representation of a cross entropy layer
internalLayer = nnet.internal.cnn.layer.CrossEntropy( ...
    args.Name, ...
    args.OutputSize);

% Pass the internal layer to a function to construct
layer = nnet.cnn.layer.ClassificationOutputLayer(internalLayer);

end

function inputArguments = iParseInputArguments(varargin)
parser = iCreateParser();
parser.parse(varargin{:});
inputArguments = iConvertToCanonicalForm(parser);
end

function p = iCreateParser()
p = inputParser;
defaultName = '';
addParameter(p, 'Name', defaultName, @iAssertValidLayerName);
end

function iAssertValidLayerName(name)
nnet.internal.cnn.layer.paramvalidation.validateLayerName(name)
end

function inputArguments = iConvertToCanonicalForm(p)
inputArguments = struct;
inputArguments.OutputSize = [];
inputArguments.Name = char(p.Results.Name); % make sure strings get converted to char vectors
end