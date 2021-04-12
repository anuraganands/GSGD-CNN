classdef MovingAverageSeries < nnet.internal.cnn.ui.axes.Series
    % MovingAverageSeries   Computes the moving average of a given Series
    
    %   Copyright 2017 The MathWorks, Inc.
    
    properties(Access = private)
        % Series   (nnet.internal.cnn.ui.axes.Series)
        Series
        
        % TrailingWindow   (integer)
        TrailingWindow
    end
    
    properties(Dependent, SetAccess = private)
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
        function this = MovingAverageSeries(series, trailingWindowSize)
            this.Series = series; 
            this.TrailingWindow = [trailingWindowSize-1, 0];
        end
        
        function xvalues = get.XValues(this)
            xvalues = this.Series.XValues;
        end
        
        function yvalues = get.YValues(this)
            yvalues = movmean(this.Series.YValues, this.TrailingWindow);
        end
        
        function xbounds = get.XBounds(this)
            xbounds = iBounds(this.XValues); 
        end
        
        function ybounds = get.YBounds(this)
            ybounds = iBounds(this.YValues);
        end
    end
end

function bounds = iBounds(values)
bounds = [min(values), max(values)];
end
