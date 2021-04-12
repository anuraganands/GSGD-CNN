classdef(Abstract) MetricRowDataFactory < handle
    % MetricRowDataFactory   Interface for returning table-row data for metric information
    
    %   Copyright 2017 The MathWorks, Inc.
    
    methods(Abstract)
        sectionStruct = createMetricRowData(~, infoStruct)
        % createMetricRowData   Given the infostruct from the
        % TrainingPlotReporter, compute a struct that corresponds to a row
        % in a TextTable. Therefore the struct must contain the following
        % fields:
        %   - RowID
        %   - LeftText
        %   - RightText
    end
    
end

