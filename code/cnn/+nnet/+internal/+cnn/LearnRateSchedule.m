classdef(Abstract) LearnRateSchedule
    % LearnRateSchedule   Abstract class for learning rate schedules
    
    %   Copyright 2015 The MathWorks, Inc.
    
    methods(Abstract)
        newLearnRate = update(this, oldLearnRate, epoch)
    end
end