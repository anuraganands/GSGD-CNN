classdef(Abstract) MarkerStrategy < handle
    % MarkerStrategy   Interface for classes which decide which points on a line should have markers
    
    %   Copyright 2017 The MathWorks, Inc.
    
    methods(Abstract)
        markerIndices = computeMarkerIndices(this, values)
        % computeMarkerIndices   Given values on a line, return an array of
        % the indices of the inputted values which should have markers.
    end
    
end

