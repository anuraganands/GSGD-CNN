function outputArray = unpadArray(inputArray, paddingSize)
% unpadArray   "Unpad" or crop a 4D array spatially
%
%   outputArray = unpadArray(inputArray, paddingSize) takes a 4D array that
%   has been spatially padded, and "unpads" it.
%
%   Inputs:
%       inputArray              - A (H+top+bottom)x(W+left+right)x(C)x(N)
%                                 array which has been padded.
%       paddingSize             - A 1-by-4 vector for the padding in the 
%                                 format [top bottom left right].
%
%   Output:
%       outputArray             - A (H)x(W)x(C)x(N) array.

%   Copyright 2017 The MathWorks, Inc.

topPad = paddingSize(1);
bottomPad = paddingSize(2);
leftPad = paddingSize(3);
rightPad = paddingSize(4);

imageTop = topPad + 1;
imageBottom = size(inputArray,1) - bottomPad;
imageLeft = leftPad + 1;
imageRight = size(inputArray,2) - rightPad;

outputArray = inputArray(imageTop:imageBottom,imageLeft:imageRight,:,:);

end