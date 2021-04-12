function validateLayerName(arg)
% validateLayerName   Throw an error if the input is not valid for use as 
% a layer name. A valid layer name must be a character row vector
% or a scalar string. Empty names ('' or "") are also allowed.

%   Copyright 2016-2017 The MathWorks, Inc.

if ~iIsValidStringOrCharArray(arg)
    error(message('nnet_cnn:layer:Layer:NameParameterIsInvalid'));
end

end

function tf = iIsValidStringOrCharArray(value)
tf = nnet.internal.cnn.layer.paramvalidation.isValidStringOrCharArray(value);
end