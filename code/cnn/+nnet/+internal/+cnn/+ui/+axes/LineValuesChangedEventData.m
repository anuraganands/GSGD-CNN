classdef(ConstructOnLoad) LineValuesChangedEventData < event.EventData
    % LineValuesChangedEventData   Class holding index of the LineModel that changed in AxesModel
    
    %   Copyright 2017 The MathWorks, Inc.
    
    properties(SetAccess = private)
        % LineIndexChanged   (integer) The index of the LineModel that
        % changed
        LineIndexChanged
    end
    
    methods
        function this = LineValuesChangedEventData(lineIndex)
            this.LineIndexChanged = lineIndex; 
        end
    end
    
end

