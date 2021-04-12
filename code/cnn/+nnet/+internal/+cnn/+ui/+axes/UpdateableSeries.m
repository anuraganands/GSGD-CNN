classdef UpdateableSeries < nnet.internal.cnn.ui.axes.Series
    % UpdateableSeries   Updateable series.
    
    %   Copyright 2017 The MathWorks, Inc.
    
    properties(SetAccess = private)
        % XValues   X-values of series
        XValues
        
        % YValues   Y-values of series
        YValues
        
        % XBounds   Bounds on the X-values
        XBounds
        
        % YBounds   Bounds on the Y-values
        YBounds
    end
    
    methods         
        function add(this, xValue, yValue)
            this.XValues = [this.XValues, xValue];
            this.YValues = [this.YValues, yValue];
            this.XBounds = iUpdateBounds(this.XBounds, xValue);
            this.YBounds = iUpdateBounds(this.YBounds, yValue);
        end
    end
    
end

function newBounds = iUpdateBounds(existingBounds, newValue)
newMinValue = min([existingBounds, newValue]);
newMaxValue = max([existingBounds, newValue]);
newBounds = [newMinValue, newMaxValue];
end
