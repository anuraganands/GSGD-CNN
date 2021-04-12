function validateStride( value )
% validateStride   Throw an error if the stride parameter is invalid.

%   Copyright 2017 The MathWorks, Inc.

validateattributes(value, {'numeric'}, ...
    {'nonempty', 'real', 'integer', 'positive'});

if ~(isscalar(value) || iIsRowVectorOfTwo(value))
    error(message('nnet_cnn:layer:Layer:ParamMustBeScalarOrPair','Stride'));
end
end

function tf = iIsRowVectorOfTwo(x)
tf = isrow(x) && numel(x)==2;
end

