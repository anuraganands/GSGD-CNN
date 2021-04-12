function out = convert2training(in)
%convert2training   Convert LearnableParameter vector to TrainingLearnableParameter.
%
%   out = prepareForTraining(in) converts a vector of LearnableParameter
%   into TrainingLearnableParameter. The input can be any subclass of
%   LearnableParameter. This is used when preparing for training.
%
%   See also: nnet.internal.cnn.layer.learnable.TrainingLearnableParameter.

%   Copyright 2017 The MathWorks, Inc.


out = nnet.internal.cnn.layer.learnable.TrainingLearnableParameter.empty();
for i = 1:numel(in)
    out(i) = nnet.internal.cnn.layer.learnable.TrainingLearnableParameter();
    out(i).Value = in(i).Value;
    out(i).LearnRateFactor = in(i).LearnRateFactor;
    out(i).L2Factor = in(i).L2Factor;
end
