function paddingMode = calculatePaddingMode(padding)
% calculatePaddingMode   Calculate the padding mode from the user input
%
%   paddingMode = calculatePaddingMode(padding) takes a user specified
%   padding option, and determines what the padding mode is (either 'same'
%   or 'manual').
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
%       paddingMode             - This will either be the character array
%                                 'same' or the character array 'manual'.

%   Copyright 2017 The MathWorks, Inc.

if(iIsTheStringSame(padding))
    % We need to convert to char in case we have a string.
    paddingMode = char(padding);
else
    paddingMode = 'manual';
end
end

function tf = iIsTheStringSame(value)
tf = nnet.internal.cnn.layer.padding.isTheStringSame(value);
end