classdef(Abstract) PreprocessingDisplayer < handle
    % PreprocessingDisplayer   Interface for strategy for displaying preprocessing information
    
    %   Copyright 2017 The MathWorks, Inc.
    
    methods(Abstract)
        displayPreprocessing(this, trainingPlotView)
        % displayPreprocessing   Displays preprocessing information
        
        hidePreprocessing(this, trainingPlotView)
        % hidePreprocessing   Hides preprocessing information
    end
    
end

