function Z = batchNormalizationForwardPredict(X, beta, gamma, epsilon, inputMean, inputVar)
% Forward batch normalization on the host, predict phase
    
%   Copyright 2016-2017 The MathWorks, Inc.

scale = gamma./sqrt(inputVar + epsilon);
offset = beta - gamma.*inputMean./sqrt(inputVar + epsilon);

Z = scale.*X + offset;

end
