classdef MarkerOnlyAtEndStrategy < nnet.internal.cnn.ui.axes.MarkerStrategy
    % MarkerOnlyAtEndStrategy   Class which always picks the last point in a line to draw a marker
    
    %   Copyright 2017 The MathWorks, Inc.
    
    methods
        function markerIndices = computeMarkerIndices(~, values)
            if isempty(values)
                markerIndices = []; 
            else
                markerIndices = numel(values);
            end
        end
    end
    
end

