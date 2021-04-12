function tf = isPaddingSymmetric(paddingSize)
% isPaddingSymmetric   Determine if padding is Symmetric or not
%
%   tf = isPaddingSymmetric(paddingSize) returns true if padding is
%   symmetric and false if it is asymmetric.
%
%   Input:
%       paddingSize             - A 1-by-4 vector for the padding in the 
%                                 format [top bottom left right].
%
%   Output:
%       tf                      - True if padding is symmetric, false if it
%                                 is asymmetric.

%   Copyright 2017 The MathWorks, Inc.

tf = ( paddingSize(1) == paddingSize(2) ) && ( paddingSize(3) == paddingSize(4) );
end