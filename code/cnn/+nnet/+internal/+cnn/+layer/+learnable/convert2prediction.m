function out = convert2prediction(in)
%convert2prediction   Convert LearnableParameter vector to PredictionLearnableParameter.
%
%   out = convert2prediction(in) converts a vector of LearnableParameter
%   into PredictionLearnableParameter. The input can be any subclass of
%   LearnableParameter. This is used when preparing for prediction.
%
%   See also: nnet.internal.cnn.layer.learnable.PredictionLearnableParameter.

%   Copyright 2017 The MathWorks, Inc.

out = nnet.internal.cnn.layer.learnable.PredictionLearnableParameter.empty();
for i = 1:numel(in)
    out(i) = nnet.internal.cnn.layer.learnable.PredictionLearnableParameter();
    out(i).Value = gather(in(i).Value);
    out(i).LearnRateFactor = in(i).LearnRateFactor;
    out(i).L2Factor = in(i).L2Factor;
    out(i).UseGPU = false;
end
