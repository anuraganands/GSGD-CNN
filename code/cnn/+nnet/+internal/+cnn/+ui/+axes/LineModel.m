classdef(Abstract) LineModel < handle
    % LineModel   Interface for models for a line component
    
    %   Copyright 2017-2018 The MathWorks, Inc.
    
    properties(Abstract)
        % Name   (char) Name of line
        Name
        
        % LineStyle   (char) Style of the line. One of '-' (default), 
        % '--' (dashed) or ':' (dotted)
        LineStyle
        
        % LineWidth   (double) Width of the line. Default is 1.3.
        LineWidth 
        
        % LineColor   (rgba-quadraplet) Color of the line e.g. [1,1,1,0.5]
        % is semi-translucent white. Default is [0,0,0,1]
        LineColor 
        
        % MarkerType   (char) Type of marker. 'o' or 'none' (default).
        MarkerType 
        
        % MarkerSize   (double) Size of marker. Default is 5
        MarkerSize 
        
        % MarkerFaceColor   (rgb-triplet) Color of the marker face. Default
        % is [0,0,0]
        MarkerFaceColor 
        
        % MarkerEdgeColor   (rgb-triplet) Color of the marker edge. Default
        % is [0,0,0]
        MarkerEdgeColor
        
        % HasFinalPointAnnotation (boolean) True if the final point should
        % be annotated.
        HasFinalPointAnnotation
    end
    
    properties(Abstract, SetAccess = private)
        % XValues   (1xn doubles) The x-values of the line
        XValues
        
        % YValues   (1xn doubles) The y-values of the line
        YValues
        
        % XBounds   (1x2 doubles) The bounds on the x-values of the line
        % expressed as [min, max]
        XBounds
        
        % YBounds   (1x2 doubles) The bounds on the y-values of the line
        % expressed as [min, max]
        YBounds
        
        % MarkerIndices   (array of indices) Indices of XValues/YValues
        % which should be adorned with markers.
        MarkerIndices
    end  
end

