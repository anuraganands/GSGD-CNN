classdef(Abstract) AxesModel < handle
    % AxesModel   Interface for models of axes
    
    %   Copyright 2017 The MathWorks, Inc.
    
    properties(Abstract, SetAccess = private)
        % XLabel   (char)
        XLabel
        
        % YLabel   (char)
        YLabel
        
        % MinXLim   (1x2 array) Any XLim must cover this MinXLim. If empty,
        % then there's no minimum x-bounds.
        MinXLim
        
        % MinYLim   (1x2 array) Any YLim must cover this MinYLim. If empty,
        % then there's no minimum y-bounds.
        MinYLim
        
        % XLim   (1x2 array) The X limits
        XLim
        
        % YLim   (1x2 array) The Y limits
        YLim
        
        % LineModels   (cell of nnet.internal.cnn.ui.axes.LineModel)
        LineModels 
    end
    
    properties(Abstract, SetAccess = private)        
        % EpochInfo   (nnet.internal.cnn.ui.EpochInfo) Information relating
        % to epochs and iterations.
        EpochInfo
        
        % EpochIndicesForTexts  (array of integers) The indices of the
        % epochs which have text labels.
        EpochIndicesForTexts
    end    
end

