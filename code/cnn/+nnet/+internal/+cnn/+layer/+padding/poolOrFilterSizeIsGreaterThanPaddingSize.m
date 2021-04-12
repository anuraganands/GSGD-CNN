function tf = poolOrFilterSizeIsGreaterThanPaddingSize(filterOrPoolSize, paddingSize)
% poolOrFilterSizeIsGreaterThanPaddingSize   Check that pool or filter size
% is greater than the padding
%
%   tf = poolOrFilterSizeIsGreaterThanPaddingSize(filterOrPoolSize, paddingSize)
%   will return a logical value which is true if the filter or pool size is
%   greater than the padding size.
%
%   Inputs:
%       filterOrPoolSize        - A 1-by-2 vector [r s] where r is the
%                                 height and s is the width of a
%                                 filter/pooling region.
%       paddingSize             - The paddingSize after the 'Padding' name 
%                                 value pair has been parsed at layer 
%                                 creation time. It can be:
%                                   - An empty array (this will be the case
%                                     if the user has specified 'same' 
%                                     padding).
%                                   - A 1-by-4 vector for the padding in 
%                                     the format [top bottom left right].
%
%   Output:
%       tf                      - Will be true if filterOrPoolSize is
%                                 greater than paddingSize in terms of both
%                                 the height and width dimensions.

%   Copyright 2017 The MathWorks, Inc.

if(isempty(paddingSize))
    tf = true;
else
    filterOrPoolSize1By4 = iExpandFilterOrPoolSize(filterOrPoolSize);
    tf = all(filterOrPoolSize1By4 > paddingSize);
end
end

function filterOrPoolSize1By4 = iExpandFilterOrPoolSize(filterOrPoolSize)
filterOrPoolSize1By4 = [filterOrPoolSize(1) filterOrPoolSize(1) filterOrPoolSize(2) filterOrPoolSize(2)];
end