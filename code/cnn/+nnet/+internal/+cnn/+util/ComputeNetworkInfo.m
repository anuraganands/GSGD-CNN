function networkInfo = ComputeNetworkInfo(seriesNetwork)
% ComputeNetworkInfo   Function that computes information from SeriesNetwork

%   Copyright 2017 The MathWorks, Inc.
    
shouldImageNormalizationBeComputed = iIsImageNormalizationToBeComputed(seriesNetwork);
networkInfo = nnet.internal.cnn.NetworkInfo(shouldImageNormalizationBeComputed);
end

% helpers
function tf = iIsImageNormalizationToBeComputed(network)
normalization = iGetNormalization(network);
zerocenter = arrayfun(@(x)isa(x, 'nnet.internal.cnn.layer.ZeroCenterImageTransform'), normalization);
assert(sum(zerocenter == 1)<=1, 'There should only be at most 1 zero center');
tf = any(zerocenter);
end
        
function n = iGetNormalization(network)
if isempty(network.Layers) || isa( network.Layers{1}, 'nnet.internal.cnn.layer.SequenceInput')
    n = nnet.internal.cnn.layer.ImageTransform.empty;
else
    n = network.Layers{1}.Transforms;
end
end