function layer = additionLayer(varargin)
% additionLayer   Addition layer
%
%   layer = additionLayer(numInputs) creates an addition layer with the
%   number of inputs specified by numInputs. This layer takes multiple
%   inputs and adds them element-wise.
%
%   layer = additionLayer(numInputs, 'PARAM1', VAL1) specifies optional parameter
%   name/value pairs for creating the layer:
%
%       'Name'                    - A name for the layer. The default is
%                                   ''.
%   An addition layer has the following inputs:
%       'in1','in2',...,'inN'     - Inputs to be added together. Note that 
%                                   all of the inputs must have the same
%                                   dimensions. See the example below for
%                                   usage.
%
%   Example:
%       Create an addition layer with two inputs that sums the output from
%       two ReLU layers.
%
%       add_1 = additionLayer(2,'Name','add_1');
%       relu_1 = reluLayer('Name','relu_1');
%       relu_2 = reluLayer('Name','relu_2');
%
%       lgraph = layerGraph();
%       lgraph = addLayers(lgraph, relu_1);
%       lgraph = addLayers(lgraph, relu_2);
%       lgraph = addLayers(lgraph, add_1);
%
%       lgraph = connectLayers(lgraph, 'relu_1', 'add_1/in1');
%       lgraph = connectLayers(lgraph, 'relu_2', 'add_1/in2');
%
%       plot(lgraph);
%
%   See also nnet.cnn.layer.AdditionLayer, depthConcatenationLayer, reluLayer.

%   Copyright 2017 The MathWorks, Inc.

% Parse the input arguments.
inputArguments = iParseInputArguments(varargin{:});

% Create an internal representation of a depth slice layer.
internalLayer = nnet.internal.cnn.layer.Addition(inputArguments.Name, inputArguments.NumInputs);

% Pass the internal layer to a function to construct a user visible depth
% slice layer.
layer = nnet.cnn.layer.AdditionLayer(internalLayer);

end

function inputArguments = iParseInputArguments(varargin)
p = inputParser;
addRequired(p, 'NumInputs', @iAssertValidNumInputs);
addParameter(p, 'Name', '', @nnet.internal.cnn.layer.paramvalidation.validateLayerName);
p.parse(varargin{:});
inputArguments = struct;
inputArguments.Name = char(p.Results.Name); % make sure strings get converted to char vectors
inputArguments.NumInputs = p.Results.NumInputs;
end

function iAssertValidNumInputs(value)
validateattributes(value, {'numeric'}, ...
    {'positive', 'real', 'integer', 'nonempty', 'scalar', '>', 1});
end
