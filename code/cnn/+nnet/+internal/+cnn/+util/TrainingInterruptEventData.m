classdef TrainingInterruptEventData < event.EventData
% TrainingInterruptEventData  Data for a TrainingInterruptEvent to add an
% exception if one occurred

    %   Copyright 2017 The MathWorks, Inc.

    properties
        Exception
    end
    
    methods
        function this = TrainingInterruptEventData( exception )
            this.Exception = exception;
        end
    end
    
end
