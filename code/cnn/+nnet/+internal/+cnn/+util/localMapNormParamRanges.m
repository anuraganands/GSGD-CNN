function [minWindowSize, maxWindowSize, minBeta, minK] = localMapNormParamRanges()
% localMapNormParamRanges   Parameter ranges for local map normalization
%
% See also: nnet.internal.cnngpu.localMapNormParamRanges. 
%
%   Copyright 2016 The MathWorks, Inc.

minWindowSize = 1;
maxWindowSize = 16;
minBeta = 0.01;
minK = 1e-5;

end