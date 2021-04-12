function heightAndWidthPadding = calculateHeightAndWidthPadding(paddingSize)
% calculateHeightAndWidthPadding   Calculate the padding along the height and width
%
%   heightAndWidthPadding = calculateHeightAndWidthPadding(paddingSize)
%   takes the 1-by-4 vector paddingSize and returns the 1-by-2 vector
%   heightAndWidthPadding.
%
%   Input:
%       paddingSize             - A 1-by-4 vector for the padding in the 
%                                 format [top bottom left right].
%
%   Output:
%       heightAndWidthPadding   - A 1-by-2 vector for the height and width
%                                 padding in the format [height width].

%   Copyright 2017 The MathWorks, Inc.

totalPaddingHeight = sum(paddingSize(1:2));
totalPaddingWidth = sum(paddingSize(3:4));
heightAndWidthPadding = [totalPaddingHeight totalPaddingWidth];
end