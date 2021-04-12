function varargout = checkLayer(layer, validInputSize, varargin)
% checkLayer   Check layer validity
%
%   checkLayer(layer, validInputSize) checks the validity of a layer using
%   generated data of size validInputSize. The layer must be of type
%   nnet.layer.Layer.
%
%   checkLayer(layer, validInputSize, 'PARAM1', VAL1, 'PARAM2', VAL2, ...)
%   specifies optional parameter name/value pairs:
%
%       'ObservationDimension' - Dimension of the data which represents
%                                 observations. The function checks the 
%                                 layer with both one and multiple 
%                                 observations if this parameter is 
%                                 specified.

%   Copyright 2017-2018 The MathWorks, Inc.

% Parse and validate input arguments
inputArguments = iParseInputArguments(layer,validInputSize,varargin{:});

% Create TestCase object
tests = nnet.checklayer.TestCase( inputArguments.Layer, ...
    inputArguments.ValidInputSize, inputArguments.ObservationDimension );

% Run tests
results = tests.run();

% Print results summary
iPrintTestSummary(results);

% Return TestResults array if requested
if nargout>=1
    varargout{1} = results;
end

end

function inputArguments = iParseInputArguments(varargin)
parser = iCreateParser();
parser.parse(varargin{:});
inputArguments = parser.Results;
end

function p = iCreateParser(varargin)
p = inputParser;
addRequired(p, 'Layer', @iAssertValidLayer);
addRequired(p, 'ValidInputSize', @iAssertValidInputSize);
addParameter(p, 'ObservationDimension', [], @iAssertValidObsDimension);
end

function iAssertValidLayer(layer)
validateattributes(layer, {'nnet.layer.Layer'}, {'scalar'});
end

function iAssertValidInputSize(validInputSize)
validateattributes(validInputSize, {'numeric'}, ...
    {'vector', 'nonempty', 'integer', 'positive'});
end

function iAssertValidObsDimension(obsDimension)
validateattributes(obsDimension, {'numeric'}, ...
    {'scalar', 'nonempty', 'integer', 'positive'});
end

function iPrintTestSummary(results)
passed = sum([results.Passed]);
failed = sum([results.Failed]);
incomplete = sum([results.Incomplete]);
testTime = sum([results.Duration]);

disp( getString(message('nnet_cnn:checkLayer:TestSummaryHeading')) )
disp( getString(message('nnet_cnn:checkLayer:TestSummaryFirstLine',passed,failed,incomplete)) )
disp( getString(message('nnet_cnn:checkLayer:TestSummarySecondLine',num2str(testTime))) )
end
