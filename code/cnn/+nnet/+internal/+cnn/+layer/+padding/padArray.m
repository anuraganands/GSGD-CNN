function outputArray = padArray(inputArray, paddingSize)
% padArray   Pad a 4D array spatially (in the first two dimensions)
%
%   outputArray = padArray(inputArray, paddingSize) takes a 4D array
%   inputArray, and a vector paddingSize, and returns a spatially padded
%   array outputArray.
%
%   Inputs:
%       inputArray              - A (H)x(W)x(C)x(N) array which will be
%                                 spatially padded along the H and W 
%                                 dimensions.
%       paddingSize             - A 1-by-4 vector for the padding in the 
%                                 format [top bottom left right].
%
%   Output:
%       outputArray             - A (H+top+bottom)x(W+left+right)x(C)x(N)
%                                 array which has been padded.

%   Copyright 2017 The MathWorks, Inc.

topPad = paddingSize(1);
bottomPad = paddingSize(2);
leftPad = paddingSize(3);
rightPad = paddingSize(4);

outputSize = size(inputArray);
outputSize(1) = outputSize(1) + topPad + bottomPad;
outputSize(2) = outputSize(2) + leftPad + rightPad;
outputArray = zeros(outputSize, 'like', inputArray);

imageTop = topPad + 1;
imageBottom = topPad + size(inputArray,1);
imageLeft = leftPad + 1;
imageRight = leftPad + size(inputArray,2);

outputArray(imageTop:imageBottom, imageLeft:imageRight, :, :) = inputArray;

end