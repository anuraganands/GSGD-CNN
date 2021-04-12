classdef LocalLinearRegressionSeries < nnet.internal.cnn.ui.axes.Series
    % LocalLinearRegressionSeries   Computes the moving average of a given Series
    
    %   Copyright 2017 The MathWorks, Inc.
    
    properties(Access = private)
        % UpdateableSeries   (nnet.internal.cnn.ui.axes.UpdateableSeries)
        UpdateableSeries
        
        % Window
        Window
        
        % CachedYValues   (array of doubles) The previously cached
        % Y-values from the underlying UpdateableSeries. Used for checking
        % if we need to recompute all the Smoothed-Y values.
        CachedYValues
        
        % CachedSmoothedYValues  (array of doubles) The smoothed version of
        % CachedYValues. It's not necessary to re-compute all smoothed
        % y-values if CachedYValues is non-empty.
        CachedSmoothedYValues
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
        function this = LocalLinearRegressionSeries(updateableSeries, window)
            this.UpdateableSeries = updateableSeries; 
            this.Window = window;
        end
        
        function xvalues = get.XValues(this)
            xvalues = this.UpdateableSeries.XValues;
        end
        
        function smoothedY = get.YValues(this)
            latestYValues = this.UpdateableSeries.YValues;
            
            % Compute only the smoothed y values corresponding to the newly
            % adding y values.
            numOldValues = numel(this.CachedYValues);
            numAddedValues = numel(latestYValues) - numel(this.CachedYValues);
            smoothedY = [this.CachedSmoothedYValues, zeros(1, numAddedValues)];
            for i=1:numAddedValues
                index = i+numOldValues;
                smoothedY(index) = iSmooth(latestYValues(1:index), this.Window);
            end
            
            smoothedY = iClipSmoothedDataToBoundsOfActualData(smoothedY, this.UpdateableSeries.YBounds);
            
            % Update cached values.
            this.CachedYValues = latestYValues;
            this.CachedSmoothedYValues = smoothedY;
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

function avgValue = iSmooth(values, window)
numValues = numel(values);
lowestIndexOfWindow = max(numValues-window+1, 1);
indicesInWindow = lowestIndexOfWindow:numValues;
valuesInWindow = values(indicesInWindow);

avgValue = iLocalLinearRegression(indicesInWindow, valuesInWindow);
end

function fittedLastYValue = iLocalLinearRegression(x,y)
if isempty(y)
    fittedLastYValue = [];
elseif isnan(y(end))
    fittedLastYValue = NaN;
else
    % Preprocess x and y.
    [x,y] = iRemoveObservationsThatHaveNaNYValues(x,y);
    x = iCenter(x);
    
    % Compute various quantities.
    variancesMatrixOfJointXandY = iVarianceMatrixForJointXandY(x, y);
    covXY = variancesMatrixOfJointXandY(2,1);
    varX = variancesMatrixOfJointXandY(1,1);
    
    % Compute fitted coeffs.
    betaHat = covXY / varX;
    alphaHat = mean(y);

    % Predict for the last y-value only.
    if varX == 0 
        fittedLastYValue = alphaHat;
    else
        fittedLastYValue = alphaHat + betaHat*(x(end));
    end
end
end

function [x,y] = iRemoveObservationsThatHaveNaNYValues(x,y)
nanY = isnan(y);
x = x(~nanY);
y = y(~nanY);
end

function x = iCenter(x)
x = x - mean(x);
end

function mat = iVarianceMatrixForJointXandY(x,y)
% for vectors x and y which are random realisations of scalar random
% variables X and Y respectively, compute the empirical variance matrix for
% the 2x1 random-variable-vector (X,Y)'. 
mat = cov(x,y,1);
end

function smoothedClippedY = iClipSmoothedDataToBoundsOfActualData(smoothedY, yBounds)
% Without clipping, the smoothed y-values can extend outside the
% bounds of the actual y-data. This can e.g. cause the accuracy
% to exceed 100 %. Therefore, clip the y-values of the smoothed result so
% that they can never lie outside the bounds of the data.

smoothedClippedY = smoothedY;

if( ~isempty(yBounds) )
    smoothedClippedY(smoothedClippedY < yBounds(1)) = yBounds(1);
    smoothedClippedY(smoothedClippedY > yBounds(2)) = yBounds(2);
end

end

