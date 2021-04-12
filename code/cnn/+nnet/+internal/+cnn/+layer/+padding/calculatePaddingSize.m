function paddingSize = calculatePaddingSize(padding)
% calculatePaddingSize   Calculate the padding size from the user input
%
%   paddingSize = calculatePaddingSize(padding) takes a user specified
%   padding option, and determines what the padding size is.
%
%   Input:
%       padding                 - This will be a user specified padding
%                                 option. It can be:
%                                   - The charater array or string 'same'.
%                                   - A single number.
%                                   - A 1-by-2 vector.
%                                   - A 1-by-4 vector.
%
%   Output:
%       paddingSize             - A 1-by-4 vector for the padding in the 
%                                 format [top bottom left right].

%   Copyright 2017 The MathWorks, Inc.

if(iIsTheStringSame(padding))
    paddingSize = [];
else
    paddingSize = iExpandPadding(padding);
end
end

function tf = iIsTheStringSame(value)
tf = nnet.internal.cnn.layer.padding.isTheStringSame(value);
end

function padding1By4 = iExpandPadding(padding)
if(isscalar(padding))
    padding1By4 = [padding padding padding padding];
elseif(iIsRowVectorOfTwo(padding))
    padding1By4 = [padding(1) padding(1) padding(2) padding(2)];
else
    padding1By4 = padding;
end
end

function tf = iIsRowVectorOfTwo(x)
tf = isrow(x) && numel(x)==2;
end