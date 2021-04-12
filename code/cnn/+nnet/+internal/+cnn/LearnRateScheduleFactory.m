classdef LearnRateScheduleFactory
    % LearnRateScheduleFactory   Factory class for creating learning rate schedules
    %
    %   This is factory class for creating learning rate schedules.
    %
    %   LearningRateSchedule methods:
    %       create                      - Static factory method for
    %                                     creating learning rate schedules.
    
    %   Copyright 2015 The MathWorks, Inc.
    
    methods (Static)
        function schedule = create(scheduleType, varargin)
            scheduleConstructors = iScheduleConstructors();
            schedule = scheduleConstructors.(scheduleType)(varargin{:});
        end
    end
end

function scheduleConstructors = iScheduleConstructors()
scheduleConstructors = struct( ...
    'none', @nnet.internal.cnn.NullSchedule, ...
    'piecewise', @nnet.internal.cnn.PiecewiseSchedule);
end