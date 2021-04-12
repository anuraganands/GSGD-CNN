function xBounded = boundAwayFromZero(x)
% boundAwayFromZero   Round any values less eps to eps
%
%   xBounded = boundAwayFromZero(x) takes an input array x and returns an
%   array xBounded where all values less than eps are replaced with eps.

%   Copyright 2016 The MathWorks, Inc.

xBounded = x;
precision = iGetPrecision(x);
xBounded(xBounded < eps(precision)) = eps(precision);
end

function precision = iGetPrecision(x)
if(isa(x,'gpuArray'))
    precision = classUnderlying(x);
else
    precision = class(x);
end
end