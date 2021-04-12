classdef MarkerOnAllPointsStrategy < nnet.internal.cnn.ui.axes.MarkerStrategy
    % MarkerOnAllPointsStrategy   Class which always picks all points in a line to draw a marker
    
    %   Copyright 2017 The MathWorks, Inc.
    
    methods
        function markerIndices = computeMarkerIndices(~, values)
            markerIndices = 1:numel(values);
        end
    end
end

