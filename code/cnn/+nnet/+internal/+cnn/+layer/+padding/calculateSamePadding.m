function paddingSize = calculateSamePadding(filterOrPoolSize, stride, inputSize)
% calculateSamePadding   Calculate 'same' padding
%
%   paddingSize = calculateSamePadding(filterOrPoolSize, stride, inputSize)
%   caluclates 'same' style padding.
%   
%   Inputs:
%       filterOrPoolSize        - A 1-by-2 vector [r s] where r is the
%                                 height and s is the width of a
%                                 filter/pooling region.
%       stride                  - A 1-by-2 vector [u v] where u is the
%                                 vertical stride and v is the horizontal 
%                                 stride.
%       inputSize               - A 1-by-2 vector [h w] where h is the
%                                 height of the input and w is the width of
%                                 the input.
%
%   Output:
%       paddingSize             - A 1-by-4 vector for the padding in the 
%                                 format [top bottom left right].

%   Copyright 2017 The MathWorks, Inc.

if(iInputSizeIsNaN(inputSize))
    % The input size will be NaN if "inferSize" has been called because a
    % user is setting a property of the layer (like the weights of a
    % convolutional layer). The padding can only be calculate at training
    % time, so we return empty.
    paddingSize = [];
else
    desiredOutputSize = ceil(inputSize./stride);
    desiredPaddedInputSize = (desiredOutputSize - 1) .* stride + filterOrPoolSize;
    totalPaddingNeeded = desiredPaddedInputSize - inputSize;
    
    % Need to ensure padding is positive.
    totalPaddingNeeded(totalPaddingNeeded < 0) = 0;
    
    topPadding = floor(totalPaddingNeeded(1)/2);
    bottomPadding = ceil(totalPaddingNeeded(1)/2);
    leftPadding = floor(totalPaddingNeeded(2)/2);
    rightPadding = ceil(totalPaddingNeeded(2)/2);
    paddingSize = [topPadding bottomPadding leftPadding rightPadding];
end

end

function tf = iInputSizeIsNaN(inputSize)
tf = all(isnan(inputSize));
end