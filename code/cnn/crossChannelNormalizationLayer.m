function layer = crossChannelNormalizationLayer( varargin )
% crossChannelNormalizationLayer   Local response normalization along channels
%
%   layer = crossChannelNormalizationLayer(windowChannelSize) creates a
%   local response normalization layer, which carries out channel-wise
%   normalization as described in [1]. For each element in the input x, we
%   compute a normalized value y using the following formula:
%
%       y = x/(K + Alpha*ss/windowChannelSize)^Beta
%
%   where ss is the sum of squares of the elements in the normalization
%   window. This function can be seen as a form of lateral inhibition
%   between channels.
%
%   The input argument windowChannelSize specifies the size of a window
%   which controls the number of channels that are used for the
%   normalization of each element. For example, if this value is 3, each
%   element will be normalized by its neighbours in the previous channel
%   and the next channel. If windowChannelSize is even, then the window
%   will be asymmetric. For example, if it is 4, each element is normalized
%   by its neighbour in the previous channel, and by its neighbours in the
%   next two channels. The value must be a positive integer.
%
%   layer = crossChannelNormalizationLayer(windowChannelSize, 'PARAM1', VAL1, 'PARAM2', VAL2, ...)
%   specifies optional parameter name/value pairs for creating the layer:
%
%       'Alpha'                   - The Alpha term in the normalization
%                                   formula. The default is 0.0001.
%       'Beta'                    - The Beta term in the normalization
%                                   formula. The default is 0.75.
%       'K'                       - The K term in the normalization
%                                   formula. The default is 2.
%       'Name'                    - A name for the layer. The default is
%                                   ''.
%
%   Example:
%       Create a local response normalization layer for channel-wise
%       normalization, where a window of 5 channels will be used to
%       normalize each element, and the additive constant for the
%       normalizer is 1.
%
%       layer = crossChannelNormalizationLayer(5, 'K', 1);
%
% [1]   A. Krizhevsky, I. Sutskever, and G. E. Hinton, "ImageNet
%       Classification with Deep Convolutional Neural Networks", in
%       Advances in Neural Information Processing Systems 25, 2012.
%
%   See also nnet.cnn.layer.CrossChannelNormalizationLayer,
%   convolution2dLayer, maxPooling2dLayer, averagePooling2dLayer.

%   Copyright 2015-2017 The MathWorks, Inc.

% Parse the input arguments.
args = iParseInputArguments(varargin{:});

% Create an internal representation of a local response normalization layer
% for channel-wise normalization.
internalLayer = nnet.internal.cnn.layer.LocalMapNorm2D( ...
    args.Name, ...
    args.WindowChannelSize, ...
    args.Alpha, ...
    args.Beta, ...
    args.K);

% Pass the internal layer to a function to construct a user visible
% local response normalization layer.
layer = nnet.cnn.layer.CrossChannelNormalizationLayer(internalLayer);

end

function inputArguments = iParseInputArguments(varargin)
parser = iCreateParser();
parser.parse(varargin{:});
inputArguments = iConvertToCanonicalForm(parser);
end

function p = iCreateParser()
p = inputParser;

defaultAlpha = 0.0001;
defaultBeta = 0.75;
defaultK = 2;
defaultName = '';

[minWindowSize, maxWindowSize, minBeta, minK] = nnet.internal.cnn.util.localMapNormParamRanges();

p.addRequired('WindowChannelSize', @(x)iAssertValidWindowSize(x,minWindowSize,maxWindowSize) )
p.addParameter('Alpha', defaultAlpha, @iAssertValidAlpha);
p.addParameter('Beta', defaultBeta, @(x)iAssertValidBeta(x,minBeta) );
p.addParameter('K', defaultK, @(x)iAssertValidK(x,minK) );
p.addParameter('Name', defaultName, @iAssertValidLayerName);
end

function iAssertValidLayerName(name)
nnet.internal.cnn.layer.paramvalidation.validateLayerName(name)
end

function iAssertValidWindowSize(value,minWindowSize,maxWindowSize)
validateattributes(value, {'numeric'}, ...
    {'scalar','integer','>=',minWindowSize,'<=',maxWindowSize});
end

function iAssertValidAlpha(value)
validateattributes(value, {'numeric'}, ...
    {'scalar','finite','real'});
end

function iAssertValidBeta(value,minBeta)
validateattributes(value, {'numeric'}, ...
    {'scalar','finite','real','>=',minBeta});
end

function iAssertValidK(value,minK)
validateattributes(value, {'numeric'}, ...
    {'scalar','finite','real','>=',minK});
end

function inputArguments = iConvertToCanonicalForm(p)
inputArguments = struct;
inputArguments.WindowChannelSize = p.Results.WindowChannelSize;
inputArguments.Alpha = p.Results.Alpha;
inputArguments.Beta = p.Results.Beta;
inputArguments.K = p.Results.K;
inputArguments.Name = char(p.Results.Name); % make sure strings get converted to char vectors
end