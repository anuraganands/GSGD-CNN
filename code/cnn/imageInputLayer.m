function layer = imageInputLayer(varargin)
% imageInputLayer   Image input layer
%
%   layer = imageInputLayer(inputSize) defines an image input layer.
%   inputSize is the size of the input images for the layer. It must be a
%   row vector of two or three numbers.
%
%   layer = imageInputLayer(inputSize, 'PARAM1', VAL1, 'PARAM2', VAL2, ...)
%   specifies optional parameter name/value pairs for creating the layer:
%
%    'DataAugmentation'  The 'DataAugmentation' parameter is not recommended. 
%                        Use augmentedImageSource instead.
%
%    'Normalization'     Specify the data normalization to apply as a
%                        string. Valid values are 'zerocenter' or 'none'.
%                        Normalization is applied every time data is
%                        forward propagated through the input layer.
%
%                        Default: 'zerocenter'
%
%    'Name'              A name for the layer.
%
%                        Default: ''
%
%   Example:
%       Create an image input layer for 28-by-28 color images. 
%
%       layer = imageInputLayer([28 28 3]);
%
%   See also nnet.cnn.layer.ImageInputLayer, convolution2dLayer,
%   fullyConnectedLayer, maxPooling2dLayer.

%   Copyright 2015-2017 The MathWorks, Inc.

% Parse the input arguments.
inputArguments = iParseInputArguments(varargin{:});

normalization = iCreateTransforms(...
    inputArguments.Normalization, inputArguments.InputSize);

augmentations = iCreateTransforms(...
    inputArguments.DataAugmentation, inputArguments.InputSize);

% Create an internal representation of an image input layer.
internalLayer = nnet.internal.cnn.layer.ImageInput(...
    inputArguments.Name, ...
    inputArguments.InputSize, ...
    normalization, ...
    augmentations);

% Pass the internal layer to a function to construct a user visible image
% input layer.
layer = nnet.cnn.layer.ImageInputLayer(internalLayer);

end

function inputArguments = iParseInputArguments(varargin)
parser = iCreateParser();
parser.parse(varargin{:});
inputArguments = iConvertToCanonicalForm(parser.Results);
end

function p = iCreateParser(varargin)
p = inputParser;

defaultName = '';
defaultTransform = 'zerocenter';
defaultTrainTransform = 'none';

addRequired(p,  'InputSize', @iAssertValidInputSize);
addParameter(p, 'Normalization', defaultTransform, @(x)any(iCheckAndReturnValidNormalization(x)));
addParameter(p, 'DataAugmentation', defaultTrainTransform, @(x)~isempty(iCheckAndReturnValidAugmentation(x)));
addParameter(p, 'Name', defaultName, @iAssertValidLayerName);
end

function iAssertValidLayerName(name)
nnet.internal.cnn.layer.paramvalidation.validateLayerName(name)
end

function iAssertValidInputSize(sz)
isValidSize = iIsRowVectorOfTwoOrThree(sz) && iIsPositiveInteger(sz) && ...
    (iIsValidGrayscaleImageSize(sz) || iIsValidRGBImageSize(sz) || iIsValidMultiChannelImageSize(sz));
if ~isValidSize
    error(message('nnet_cnn:layer:ImageInputLayer:InvalidImageSize'));
end
end

function inputArguments = iConvertToCanonicalForm(params)
try
    inputArguments = struct;
    inputArguments.InputSize = iMakeIntoRowVectorOfThree(params.InputSize);
    inputArguments.Normalization = iCheckAndReturnValidNormalization(params.Normalization);
    inputArguments.DataAugmentation = iCheckAndReturnValidAugmentation(params.DataAugmentation);
    inputArguments.Name = char(params.Name); % make sure strings get converted to char vectors
catch e
    % Reduce the stack trace of the error message by throwing as caller
    throwAsCaller(e)
end
end

function validSize = iMakeIntoRowVectorOfThree(inputSize)
if(iIsRowVectorOfTwo(inputSize))
    validSize = [inputSize 1];
else
    validSize = inputSize;
end
end

function x = iCheckAndReturnValidNormalization(x)
validTransforms = {'zerocenter', 'none'};
x = validatestring(x, validTransforms);
end

function x = iCheckAndReturnValidAugmentation(x)
validTransforms = {'randcrop', 'randfliplr', 'none'};
x = iIsValidCellStringOrStringArg(x, validTransforms);

if iscellstr(x)
    if numel(x) > 1 && iAreNotUniqueStrings(x)
        error(message('nnet_cnn:layer:ImageInputLayer:AugmentationsMustBeUnique'));
    end
        
    if numel(x) > 1 && ismember('none', x)
        error(message('nnet_cnn:layer:ImageInputLayer:NoneNotAllowedWithOthers'));
    end

    if numel(x) == 1
        % return string if only single cell element.
        x = x{1};
    end
end
end

function x = iIsValidCellStringOrStringArg(x, validstr)
if iscellstr(x) && numel(x) > 0
    x = cellfun(@(str)validatestring(str, validstr), ...
        x, 'UniformOutput', false);
else
    x = validatestring(x, validstr);
end
end

function tf = iIsValidGrayscaleImageSize(sz)
% Return true if size is [M N] or [M N 1]. Assumes input sz is already
% validated as a 2 or 3 element vector.
if numel(sz) == 2
    tf = true;
else
    tf = sz(end) == 1;
end
end

function tf = iIsValidRGBImageSize(sz)
tf = numel(sz) == 3 && sz(end) == 3;
end

function tf = iIsValidMultiChannelImageSize(sz)
% Placeholder, if more constraints need to be added
tf = numel(sz) == 3;
end

function tf = iIsRowVectorOfTwoOrThree(x)
tf = (iIsRowVectorOfTwo(x) || iIsRowVectorOfThree(x));
end

function tf = iIsPositiveInteger(x)
tf =  all(x > 0) && isreal(x) && all(mod(x,1)==0);
end

function tf = iAreNotUniqueStrings(x)
tf = numel(unique(x)) ~= numel(x);
end

function tf = iIsRowVectorOfTwo(x)
tf = isrow(x) && numel(x)==2;
end

function tf = iIsRowVectorOfThree(x)
tf = isvector(x) && all(size(x) == [1 3]);
end

function tformarray = iCreateTransforms(type, imageSize)
type = cellstr(type);
tformarray = nnet.internal.cnn.layer.ImageTransform.empty();
for i = 1:numel(type)
    tnew = nnet.internal.cnn.layer.ImageTransformFactory.create(type{i}, imageSize);
    tformarray = [tformarray tnew]; %#ok<AGROW>
end
end
