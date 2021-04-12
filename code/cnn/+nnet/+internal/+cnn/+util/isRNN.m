function tf = isRNN( layers )
% isRNN   Determine if a layer array defines a recurrent neural network.

%   Copyright 2017 The MathWorks, Inc.

tf = false;
if ~isempty( layers )
    tf = isa(layers{1}, 'nnet.internal.cnn.layer.SequenceInput');
end
end