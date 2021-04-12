function tf = isValidStringOrCharArray( value )
% isValidStringOrCharArray   User input for options that are 'strings' must
% be a character row vector or a scalar string. Empty names ('' or "") are 
% also allowed.

%   Copyright 2017 The MathWorks, Inc.

isCharRowVectorOrEmtpy = ischar(value) && (isrow(value) || isempty(value));
isScalarString = (isstring(value) && isscalar(value));
tf = isCharRowVectorOrEmtpy || isScalarString;
end