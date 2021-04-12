function validatePadding( value )
% validatePooling   Throw an error if the padding parameter is invalid.

%   Copyright 2017 The MathWorks, Inc.

if iIsValidStringOrCharArray(value)
    validatestring(value,{'same'});
else
    validateattributes(value, {'numeric'}, ...
    	{'nonempty', 'real', 'integer', 'nonnegative'});

    if ~(isscalar(value) || iIsRowVectorOfTwo(value) || iIsRowVectorOfFour(value))
        error(message('nnet_cnn:layer:Layer:ParamMustBeScalarOrTwoOrFour','Padding'));
    end
end

end

function tf = iIsValidStringOrCharArray(value)
tf = nnet.internal.cnn.layer.paramvalidation.isValidStringOrCharArray(value);
end

function tf = iIsRowVectorOfTwo(x)
tf = isrow(x) && numel(x)==2;
end

function tf = iIsRowVectorOfFour(x)
tf = isrow(x) && numel(x)==4;
end