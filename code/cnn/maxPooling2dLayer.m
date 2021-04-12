function layer = maxPooling2dLayer( varargin )
% maxPooling2dLayer   Max pooling layer
%
%   layer = maxPooling2dLayer(poolSize) creates a layer that performs max
%   pooling. A max pooling layer divides the input into rectangular pooling
%   regions, and outputs the maximum of each region. poolSize specifies the
%   width and height of a pooling region. It can be a scalar, in which case
%   the pooling regions will have the same width and height, or a vector
%   [h w] where h specifies the height and w specifies the width. Note that
%   if the 'Stride' dimensions are less than the respective pool
%   dimensions, then the pooling regions will overlap.
%
%   This layer also outputs indices of the maximum values in each pooled
%   region and the size of the input feature map. These outputs are only
%   supported when the pooling regions do not overlap.
%
%   layer = maxPooling2dLayer(poolSize, 'PARAM1', VAL1, 'PARAM2', VAL2, ...)
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
%       'HasUnpoolingOutputs'        - Specifies whether this layer should
%                                   have extra outputs that can be used for
%                                   unpooling.
%                                     - If this is false, then the layer
%                                       has a single output with the name
%                                       'out'
%                                     - If this is true, then the layer has
%                                       two additional outputs with the
%                                       names 'indices' and 'size' that you
%                                       can connect to a max unpooling
%                                       layer. See example 3 below for
%                                       details.
%                                   The default is false.
%       'Name'                    - A name for the layer. The default is
%                                   ''.
%
%   A max pooling layer has the following outputs:
%       'out'     - Pooled output feature maps.
%       'indices' - The indices of the maximum value in each pooled region.
%                   Will only be present if 'HasUnpoolingOutputs' is true.
%       'size'    - Size of the input feature map. Will only be present if
%                   'HasUnpoolingOutputs' is true.
%
%   Example 1:
%       Create a max pooling layer with non-overlapping pooling regions,
%       which downsamples by a factor of 2.
%
%       layer = maxPooling2dLayer(2, 'Stride', 2);
%
%   Example 2:
%       Create a max pooling layer with overlapping pooling regions and
%       padding for the top and bottom of the input.
%
%       layer = maxPooling2dLayer(2, 'Stride', 2, 'Padding', [1 0]);
%
%   Example 3:
%       Unpool the output of max pooling layer by connecting the max
%       pooling layer to the max unpooling layer.
%
%       layers = [
%            maxPooling2dLayer(2, 'Stride', 2, 'HasUnpoolingOutputs', true, 'Name', 'mpool')
%            maxUnpooling2dLayer('Name', 'unpool')];
%
%       % Sequentially connect layers by adding them to a layerGraph. This
%       % connects max pooling layer's 'out' to max unpooling layer's 'in'.
%       lgraph = layerGraph(layers)
%
%       % Connect optional max pooling layer outputs to unpooling layer inputs.
%       lgraph = connectLayers(lgraph, 'mpool/indices', 'unpool/indices');
%       lgraph = connectLayers(lgraph, 'mpool/size', 'unpool/size');
%
%   See also nnet.cnn.layer.MaxPooling2DLayer, averagePooling2dLayer,
%            maxUnpooling2dLayer, convolution2dLayer, LayerGraph,
%            LayerGraph/connectLayers, LayerGraph/plot.

%   Copyright 2015-2017 The MathWorks, Inc.

% Parse the input arguments.
inputArguments = iParseInputArguments(varargin{:});

% Create an internal representation of a max pooling layer.
internalLayer = nnet.internal.cnn.layer.MaxPooling2D( ...
    inputArguments.Name, ...
    inputArguments.PoolSize, ...
    inputArguments.Stride, ...
    inputArguments.PaddingMode, ...
    inputArguments.PaddingSize, ...
    inputArguments.HasUnpoolingOutputs );

% Pass the internal layer to a function to construct a user visible
% max pooling layer
layer = nnet.cnn.layer.MaxPooling2DLayer(internalLayer);

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
defaultUnpoolingOutputs = false;
defaultName = '';

addRequired(p, 'PoolSize', @iAssertValidPoolSize);
addParameter(p, 'Stride', defaultStride, @iAssertValidStride);
addParameter(p, 'Padding', defaultPadding, @iAssertValidPadding);
addParameter(p, 'HasUnpoolingOutputs', defaultUnpoolingOutputs, @iAssertValidUnpoolingOutputs);
addParameter(p, 'Name', defaultName, @iAssertValidLayerName);
end

function inputArguments = iConvertToCanonicalForm(params)
inputArguments = struct;
inputArguments.PoolSize = iMakeIntoRowVectorOfTwo(params.PoolSize);
inputArguments.Stride = iMakeIntoRowVectorOfTwo(params.Stride);
inputArguments.PaddingMode = iCalculatePaddingMode(params.Padding);
inputArguments.PaddingSize = iCalculatePaddingSize(params.Padding);
inputArguments.Name = char(params.Name); % make sure strings get converted to char vectors
inputArguments.HasUnpoolingOutputs = params.HasUnpoolingOutputs;
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

function iAssertValidUnpoolingOutputs(value)
validateattributes(value, {'logical'}, {'scalar'});
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
    exception = MException(message('nnet_cnn:layer:MaxPooling2DLayer:PaddingSizeLargerThanOrEqualToPoolSize'));
    throwAsCaller(exception);
end
end

function tf = iPoolSizeIsGreaterThanPaddingSize(poolSize, paddingSize)
tf = nnet.internal.cnn.layer.padding.poolOrFilterSizeIsGreaterThanPaddingSize(poolSize, paddingSize);
end