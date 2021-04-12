classdef(Abstract) Series < handle
    % Series   Interface for a whole series of values
    
    %   Copyright 2017 The MathWorks, Inc.
    
    properties(Abstract, SetAccess = private)
        % XValues   X-values of series
        XValues
        
        % YValues   Y-values of series
        YValues
        
        % XBounds   Bounds on the X-values
        XBounds
        
        % YBounds   Bounds on the Y-values
        YBounds
    end
    
end

