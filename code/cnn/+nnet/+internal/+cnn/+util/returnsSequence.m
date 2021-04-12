function [returnSeq, statefulLayers] = returnsSequence( internalLayers, isRNN )
% returnsSequence   Determine if an RNN layer array is configured such that
% its output is a sequence, or a single element.

%   Copyright 2017 The MathWorks, Inc.

% Determine if network will return sequence
returnSeq = false;
numLayers = numel( internalLayers );
statefulLayers = false( numLayers, 1 );
if isRNN
    layerReturnsSequence = logical.empty();
    for ii = 1:numLayers
        if isa( internalLayers{ii}, 'nnet.internal.cnn.layer.Updatable' )
            layerReturnsSequence = [layerReturnsSequence internalLayers{ii}.ReturnSequence]; %#ok<AGROW>
            statefulLayers(ii) = true;
        end
    end
    returnSeq = all( layerReturnsSequence );
end
end