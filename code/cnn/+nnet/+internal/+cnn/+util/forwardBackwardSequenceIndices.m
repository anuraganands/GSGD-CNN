function [fInd, bInd] = forwardBackwardSequenceIndices(HiddenSize)
% forwardBackwardSequenceIndices   Determine indices of the forward
% sequence and backward sequence of the bidirectional LSTM layer

%   Copyright 2017 The MathWorks, Inc.

fInd = 1:4*HiddenSize;
bInd = 4*HiddenSize + fInd;

end