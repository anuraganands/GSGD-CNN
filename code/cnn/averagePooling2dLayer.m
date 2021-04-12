function layer = averagePooling2dLayer( varargin )
% averagePooling2dLayer   Average pooling layer
%
%   layer = averagePooling2dLayer(poolSize) creates a layer that performs
%   average pooling. An average pooling layer divides the input into
%   rectangular pooling regions, and outputs the average of each region.
%   poolSize specifies the width and height of a pooling region. It can be
%   a scalar, in which case the pooling regions will have the same width
%   and height, or a vector [h w] where h specifies the height and w
%   specifies the width. Note that if the 'Stride' dimensions are less than
%   the respective pool dimensions, then the pooling regions will overlap.
%
%   layer = averagePooling2dLayer(poolSize, 'PARAM1', VAL1, 'PARAM2', VAL2, ...)
%   specifies optional parameter name/value pairs for creating the layer:
%
%       'Stride'                  - The step size for traversing the input
%                                   vertically and horizontally. This can
%                                   be a scalar, in which case the same
%                                   value is used for both dimensions, or
%                                   it can be a vector [u v] where u is the
%                                   vertical stride, and v is the
%                                   horizontal stride. Values that are
%                                   greater than 1 can be used to
%                                   down-sample the input. The default is
%                                   [1 1].
%       'Padding'                 - The padding applied to the input
%                                   along the edges. This can be:
%                                     - the character array 'same'. Padding
%                                       is set so that the output size 
%                                       is the same as the input size 
%                                       when the stride is 1. More 
%                                       generally, the output size is 
%                                       ceil(inputSize/stride), where 
%                                       inputSize is the height and width 
%                                       of the input.
%                                     - a scalar, in which case the same
%                                       padding is applied vertically and
%                                       horizontally.
%                                     - a vector [a b] where a is the 
%                                       padding applied to the top and 
%                                       bottom of the input, and b is the
%                                       padding applied to the left and 
%                                       right.
%                                     - a vector [t b l r] where t is the
%                                       padding applied to the top, b is
%                                       the padding applied to the bottom,
%                                       l is the padding applied to the 
%                                       left, and r is the padding applied 
%                                       to the right.
%                                   Note that the padding dimensions must
%                                   be less than the pooling region
%                                   dimensions. The default is 0.
%       'Name'                    - A name for the layer. The default is
%                                   ''.
%
%   Example 1:
%       Create an average pooling layer with non-overlapping pooling
%       regions, which downsamples by a factor of 2:
%
%       layer = averagePooling2dLayer(2, 'Stride', 2);
%
%   Example 2:
%       Create an average pooling layer with overlapping pooling regions
%       and padding for the top and bottom of the input:
%
%       layer = averagePooling2dLayer(3, 'Stride', 2, 'Padding', [1 0]);
%
%   See also nnet.cnn.layer.AveragePooling2DLayer, maxPooling2dLayer,
%   convolution2dLayer.

%   Copyright 2015-2017 The MathWorks, Inc.

% Parse the input arguments.
inputArguments = iParseInputArguments(varargin{:});

% Create an internal representation of an average pooling layer.
internalLayer = nnet.internal.cnn.layer.AveragePooling2D( ...
    inputArguments.Name, ...
    inputArguments.PoolSize, ...
    inputArguments.Stride, ...
    inputArguments.PaddingMode, ...
    inputArguments.PaddingSize);

% Pass the internal layer to a function to construct a user visible
% average pooling layer
layer = nnet.cnn.layer.AveragePooling2DLayer(internalLayer);

end

function inputArguments = iParseInputArguments(varargin)
parser = iCreateParser();
parser.parse(varargin{:});
inputArguments = iConvertToCanonicalForm(parser.Results);
iAssertPoolSizeIsGreaterThanPaddingSize(inputArguments);
end

function p = iCreateParser()
p = inputParser;
defaultStride = 1;
defaultPadding = 0;
defaultName = '';

addRequired(p, 'PoolSize', @iAssertValidPoolSize);
addParameter(p, 'Stride', defaultStride, @iAssertValidStride);
addParameter(p, 'Padding', defaultPadding, @iAssertValidPadding);
addParameter(p, 'Name', defaultName, @iAssertValidLayerName);
end

function inputArguments = iConvertToCanonicalForm(params)
inputArguments = struct;
inputArguments.PoolSize = iMakeIntoRowVectorOfTwo(params.PoolSize);
inputArguments.Stride = iMakeIntoRowVectorOfTwo(params.Stride);
inputArguments.PaddingMode = iCalculatePaddingMode(params.Padding);
inputArguments.PaddingSize = iCalculatePaddingSize(params.Padding);
inputArguments.Name = char(params.Name); % make sure strings get converted to char vectors
end

function iAssertValidLayerName(name)
nnet.internal.cnn.layer.paramvalidation.validateLayerName(name);
end

function iAssertValidPoolSize(value)
validateattributes(value, {'numeric'}, ...
    {'positive', 'real', 'integer', 'nonempty'});
iAssertScalarOrRowVectorOfTwo(value,'poolSize');
end

function iAssertValidStride(value)
validateattributes(value, {'numeric'}, ...
    {'positive', 'real', 'integer', 'nonempty'});
iAssertScalarOrRowVectorOfTwo(value,'Stride');
end

function iAssertValidPadding(value)
nnet.internal.cnn.layer.paramvalidation.validatePadding(value);
end

function iAssertScalarOrRowVectorOfTwo(value,name)
if ~(isscalar(value) || iIsRowVectorOfTwo(value))
    error(message('nnet_cnn:layer:Layer:ParamMustBeScalarOrPair',name));
end
end

function rowVectorOfTwo = iMakeIntoRowVectorOfTwo(scalarOrRowVectorOfTwo)
if(iIsRowVectorOfTwo(scalarOrRowVectorOfTwo))
    rowVectorOfTwo = scalarOrRowVectorOfTwo;
else
    rowVectorOfTwo = [scalarOrRowVectorOfTwo scalarOrRowVectorOfTwo];
end
end

function tf = iIsRowVectorOfTwo(x)
tf = isrow(x) && numel(x)==2;
end

function paddingMode = iCalculatePaddingMode(padding)
paddingMode = nnet.internal.cnn.layer.padding.calculatePaddingMode(padding);
end

function paddingSize = iCalculatePaddingSize(padding)
paddingSize = nnet.internal.cnn.layer.padding.calculatePaddingSize(padding);
end

function iAssertPoolSizeIsGreaterThanPaddingSize(inputArguments)
if(~iPoolSizeIsGreaterThanPaddingSize(inputArguments.PoolSize, inputArguments.PaddingSize))
    exception = MException(message('nnet_cnn:layer:AveragePooling2DLayer:PaddingSizeLargerThanOrEqualToPoolSize'));
    throwAsCaller(exception);
end
end

function tf = iPoolSizeIsGreaterThanPaddingSize(poolSize, paddingSize)
tf = nnet.internal.cnn.layer.padding.poolOrFilterSizeIsGreaterThanPaddingSize(poolSize, paddingSize);
end