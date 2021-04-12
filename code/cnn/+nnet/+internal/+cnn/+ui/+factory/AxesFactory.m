classdef(Abstract) AxesFactory < handle
    % AxesFactory   Interface for factories producing AxesViews and Metrics
    
    %   Copyright 2017 The MathWorks, Inc.
    
    methods(Abstract)
        [axesView, metrics] = createMainAxesAndMetrics(this, epochInfo, epochDisplayer)
        [axesView, metrics] = createLossAxesAndMetrics(this, epochInfo, epochDisplayer)
        
        [cellOfLegendSectionNames, cellOfLegendSectionStructArrs] = createLegendInfo(this);
    end
    
end

 
 

