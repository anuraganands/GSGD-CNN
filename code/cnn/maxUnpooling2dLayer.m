function layer = maxUnpooling2dLayer( varargin )
% maxUnpooling2dLayer   Max unpooling layer
%
%   layer = maxUnpooling2dLayer() creates a layer that unpools the
%   output of a max pooling layer.
%
%   layer = maxUnpooling2dLayer(..., Name, Value) specifies optional
%   parameter name-value pairs for creating the layer:
%
%       'Name'                    - A name for the layer. The default is
%                                   ''.
%
%   This layer requires three inputs. The inputs can be connected using 
%   <a href="matlab:help('layerGraph.connectLayers')">connectLayers</a>. See the example below for details.
%
%   maxUnpooling2dLayer Inputs:
%       'in'      - Input feature map to unpool.
%       'indices' - The indices of the maximum value in each pooled region. 
%                   This is output by the max pooling layer. 
%       'size'    - Output size of unpooled feature map. This is output by
%                   the max pooling layer.
%
%   Example:
%       Unpool the output of max pooling layer by connecting the max
%       pooling layer to the max unpooling layer.
%
%       layers = [
%            maxPooling2dLayer(2, 'Stride', 2, 'Name', 'mpool')
%            maxUnpooling2dLayer('Name', 'unpool');
%           ]
%       
%       % Sequentially connect layers by adding them to a layerGraph. This
%       % connects max pooling layer's 'Z' to max unpooling layer's 'X'.
%       lgraph = layerGraph(layers)
% 
%       % Connect max pooling layer outputs to unpooling layer inputs.
%       lgraph = connectLayers(lgraph, 'mpool/indices', 'unpool/indices');
%       lgraph = connectLayers(lgraph, 'mpool/size', 'unpool/size');
%
%   See also nnet.cnn.layer.MaxUnpooling2DLayer, maxPooling2dLayer, 
%            layerGraph.

%   Copyright 2017 The MathWorks, Inc.

% Parse the input arguments.
inputArguments = iParseInputArguments(varargin{:});

% Create an internal representation of a max pooling layer.
internalLayer = nnet.internal.cnn.layer.MaxUnpooling2D( ...
    inputArguments.Name);

% Pass the internal layer to a function to construct a user visible
% max pooling layer
layer = nnet.cnn.layer.MaxUnpooling2DLayer(internalLayer);
end

function inputArguments = iParseInputArguments(varargin)
parser = iCreateParser();
parser.parse(varargin{:});
userInput = parser.Results;

inputArguments = iConvertToCanonicalForm(userInput);
end

function p = iCreateParser()
p = inputParser;
defaultName = '';

addParameter(p, 'Name', defaultName, @iAssertValidLayerName);
end

function inputArguments = iConvertToCanonicalForm(params)
inputArguments = struct;
inputArguments.Name = char(params.Name); % make sure strings get converted to char vectors
end

function iAssertValidLayerName(name)
nnet.internal.cnn.layer.paramvalidation.validateLayerName(name);
end
