classdef(Abstract) EpochDisplayer < handle
    % EpochDisplayer   Interface for classes which help display epochs
    
    %   Copyright 2017 The MathWorks, Inc.
    
    methods(Abstract)
        updateEpochRectangles(this, patchObj, numEpochs, numItersPerEpoch, yBounds)
        % updateEpochRectangles   Given a patch object, update it to show
        % the epoch rectangles
        
        initializeEpochTexts(this, epochTextObjects)
        % initializeEpochTexts   Initialize the given array of text objects
        
        updateEpochTexts(this, epochTextObjects, epochIndices, numItersPerEpoch, xBounds, yBounds)
        % updateEpochTexts   Given an array of text objects, update them
        % with the given epoch texts, positions and colors.
    end
    
end

