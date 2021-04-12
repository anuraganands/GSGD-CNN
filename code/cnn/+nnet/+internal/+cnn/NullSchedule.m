classdef NullSchedule < nnet.internal.cnn.LearnRateSchedule
    % NullSchedule   A learning rate schedule that does not alter the
    % learning rate.
    
    %   Copyright 2015 The MathWorks, Inc.
    
    methods
        function this = NullSchedule()
        end
        
        function learnRate = update(~, learnRate, ~)
            % update Null schedule never changes the learning rate
        end
    end
end