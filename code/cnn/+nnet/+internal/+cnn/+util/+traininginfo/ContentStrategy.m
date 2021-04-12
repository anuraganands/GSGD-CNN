classdef (Abstract) ContentStrategy
    % ContentStrategy   Training info content strategy interface
    
    %   Copyright 2016 The MathWorks, Inc.
    
    properties (Abstract)
        % FieldNames (cellstr)   Field names to assign to the training info
        % structure
        FieldNames
        
        % SummaryNames (cellstr)   Names of MiniBatchSummary properties to 
        % be reported
        SummaryNames
    end
end