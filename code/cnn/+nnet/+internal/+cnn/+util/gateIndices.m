function [zInd, iInd, fInd, oInd] = gateIndices(HiddenSize)
% gateIndices   Determine indices of the data input, input, forget and
% output gates of the LSTM layer

%   Copyright 2017 The MathWorks, Inc.

iInd = 1:HiddenSize;
fInd = 1 + HiddenSize:2*HiddenSize;
zInd = 1 + 2*HiddenSize:3*HiddenSize;
oInd = 1 + 3*HiddenSize:4*HiddenSize;

end