classdef NoMarkerStrategy < nnet.internal.cnn.ui.axes.MarkerStrategy
    % NoMarkerStrategy   Class which always specifies that each point should have no marker.
    
    %   Copyright 2017 The MathWorks, Inc.
    
    methods
        function markerIndices = computeMarkerIndices(~, ~)
            markerIndices = [];
        end
    end
    
end

