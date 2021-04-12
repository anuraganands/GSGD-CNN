function validateLearnFactor( value, factorName )
% validateLearnFactor   Throw an error if the learnable parameter factor is
% invalid.

%   Copyright 2017 The MathWorks, Inc.

if nargin<2
    factorName = '';
end

validateattributes(value, {'numeric'}, ...
    {'scalar','real','finite','nonnegative'},...
    '', factorName);
end

