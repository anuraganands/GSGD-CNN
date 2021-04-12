function tf = isTheStringSame( value )
% isSameString   Return true if the input is the string/char array 'same'

%   Copyright 2017 The MathWorks, Inc.

if iIisValidStringOrCharArray(value)
    tf = strcmp(value, 'same');
else
    tf = false;
end
end

function tf = iIisValidStringOrCharArray(value)
tf = nnet.internal.cnn.layer.paramvalidation.isValidStringOrCharArray(value);
end