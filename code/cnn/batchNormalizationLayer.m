function layer = batchNormalizationLayer(varargin)
%batchNormalizationLayer   Batch normalization layer
%
%   layer = batchNormalizationLayer() creates a batch normalization layer.
%   This type of layer normalizes each channel across a mini-batch. This
%   can be useful in reducing sensitivity to variations within the data.
%
%   layer = batchNormalizationLayer('PARAM1', VAL1) specifies optional
%   parameter name/value pairs for creating the layer: 
%     'Name'        - A name for the layer. Default is ''. 
%     'Offset'      - Initial value for learnable parameter Offset (also
%                     called beta). Default is all zeros (1x1xNumChannels).
%     'Scale'       - Initial value for learnable parameter Scale (also
%                     called gamma). Default is all ones (1x1xNumChannels).
%     'Epsilon'     - Offset for the variance to avoid divide-by-zero
%                     errors. Must be at least 1e-5. Default is 1e-5.
%     'OffsetLearnRateFactor' - Multiplier for the learning rate of Offset.
%                               Default is 1. 
%     'ScaleLearnRateFactor'  - Multiplier for the learning rate of Scale.
%                               Default is 1.
%     'OffsetL2Factor'        - Multiplier for the L2 weight regulariser
%                               for Offset. Default is 1. 
%     'ScaleL2Factor'         - Multiplier for the L2 weight regulariser 
%                               for Scale. Default is 1. 
%
%   Example:
%     Create a batch normalization layer.
%
%     layer = batchNormalizationLayer();
%
%   See also nnet.cnn.layer.BatchNormalizationLayer.

%   Copyright 2017 The MathWorks, Inc.

% Parse the input arguments.
args = iParseInputArguments(varargin{:});

% Create an internal representation of the layer.
internalLayer = nnet.internal.cnn.layer.BatchNormalization(args.Name, [], args.Epsilon);

internalLayer.Offset.L2Factor = args.OffsetL2Factor;
internalLayer.Offset.LearnRateFactor = args.OffsetLearnRateFactor;

internalLayer.Scale.L2Factor = args.ScaleL2Factor;
internalLayer.Scale.LearnRateFactor = args.ScaleLearnRateFactor;


% Use the internal layer to construct a user visible layer.
layer = nnet.cnn.layer.BatchNormalizationLayer(internalLayer);

% Set the offset and scale (if supplied) using the visible layer so that
% the number of channels is correctly checked.
if ~isempty(args.Offset)
    layer.Offset = args.Offset;
end
if ~isempty(args.Scale)
    layer.Scale = args.Scale;
end

end

function inputArguments = iParseInputArguments(varargin)
parser = iCreateParser();
parser.parse(varargin{:});
inputArguments = parser.Results;
end

function p = iCreateParser()
p = inputParser;

minEpsilon = 1e-5; % Defined by cuDNN
defaultEpsilon = minEpsilon;
defaultOffset = [];
defaultScale = [];
defaultLearnRateFactor = 1;
defaultL2Factor = 1;
defaultName = '';

p.addParameter('Name', defaultName, @iAssertValidLayerName);
p.addParameter('Epsilon', defaultEpsilon, @(x)iAssertValidEpsilon(x,minEpsilon));
p.addParameter('Offset', defaultOffset, @iAssertValidOffset);
p.addParameter('Scale', defaultScale, @iAssertValidScale);
p.addParameter('OffsetLearnRateFactor', defaultLearnRateFactor, @iAssertValidFactor);
p.addParameter('ScaleLearnRateFactor', defaultLearnRateFactor, @iAssertValidFactor);
p.addParameter('OffsetL2Factor', defaultL2Factor, @iAssertValidFactor);
p.addParameter('ScaleL2Factor', defaultL2Factor, @iAssertValidFactor);
end

function iAssertValidLayerName(name)
nnet.internal.cnn.layer.paramvalidation.validateLayerName(name)
end

function iAssertValidEpsilon(value,minValue)
validateattributes(value, {'numeric'}, ...
    {'scalar','finite','real','nonnegative','>=',minValue});
end

function iAssertValidOffset(value)
validateattributes(value, {'numeric'}, ...
    {'finite','real'});
end

function iAssertValidScale(value)
validateattributes(value, {'numeric'}, ...
    {'finite','real'});
end

function iAssertValidFactor(value)
nnet.internal.cnn.layer.paramvalidation.validateLearnFactor(value);
end
