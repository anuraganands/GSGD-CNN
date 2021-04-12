classdef MultilineAxesModel < nnet.internal.cnn.ui.axes.AxesModel
    % MultilineAxesModel   Model for an axes object that handles multiple lines.
    
    %   Copyright 2017 The MathWorks, Inc.
    
    properties(SetAccess = private)
        % XLabel   (char)
        XLabel
        
        % YLabel   (char)
        YLabel
        
        % MinXLim   (1x2 array) Any XLim must cover this MinXLim. If empty,
        % then there's no minimum x-bounds.
        MinXLim
        
        % MinYLim   (1x2 array) Any YLim must cover this MinYLim. If empty,
        % then there's no minimum y-bounds.
        MinYLim
        
        % LineModels   (cell of nnet.internal.cnn.ui.LineModel)
        LineModels
    end
    
    properties(Dependent, SetAccess = private)
        % XLim   (1x2 array) The X limits
        XLim
        
        % YLim   (1x2 array) The Y limits
        YLim
    end
    
    properties(SetAccess = private)
        % EpochInfo   (nnet.internal.cnn.ui.EpochInfo) Information relating
        % to epochs and iterations.
        EpochInfo
    end
    
    properties(Dependent, SetAccess = private)
        % NumItersSoFar   (integer) The number of iterations computed so
        % far.
        NumItersSoFar
        
        % EpochIndicesForTexts  (array of integers) The indices of the
        % epochs which have text labels.
        EpochIndicesForTexts
    end
    
    methods
        function this = MultilineAxesModel(lineModels, xLabel, yLabel, minXLim, minYLim, epochInfo)
            this.LineModels = lineModels;
            this.XLabel = xLabel;
            this.YLabel = yLabel;
            this.MinXLim = minXLim;
            this.MinYLim = minYLim;
            this.EpochInfo = epochInfo;
        end
        
        function xlim = get.XLim(this)
            % Set of bounds from lineModels only
            xBoundsFromLines = cellfun(@(lineModel) lineModel.XBounds, this.LineModels, 'UniformOutput', false);
            
            % Include MinXLim into the bounds.
            xBounds = [xBoundsFromLines, {this.MinXLim}];
            
            xlim = iComputeBoundsThatCoverSetOfBounds(xBounds);
            xlim = iUseDefaultBoundsIfEmpty(xlim);
            xlim = iAddPaddingToBoundsIfZeroRange(xlim);
        end
        
        function ylim = get.YLim(this)
            % Set of bounds from lineModels only
            yBoundsFromLines = cellfun(@(lineModel) lineModel.YBounds, this.LineModels, 'UniformOutput', false);
            ylimFromLines = iComputeBoundsThatCoverSetOfBounds(yBoundsFromLines);
            % Add padding before we include MinXLim
            ylimFromLines = iAddYPadding(ylimFromLines);
            
            % Include MinXLim into the bounds
            yBounds = [ylimFromLines, {this.MinYLim}];
            
            ylim = iComputeBoundsThatCoverSetOfBounds(yBounds);
            ylim = iUseDefaultBoundsIfEmpty(ylim);
            ylim = iAddPaddingToBoundsIfZeroRange(ylim);
        end
        
        function epochIndices = get.EpochIndicesForTexts(this)
            numVisibleEpochs = this.computeNumVisibleEpochs();
            
            % For the given numVisibleEpochs, compute the current regime of
            % how the epochs will be displayed. For example, if
            % numVisibleEpochs==9, then we are in the regime where we label
            % every epoch, or if numVisibleEpochs==11, we label every 10
            % epochs. Note that we only need to create 10 labels because
            % the user can only see at most 10 labels. For example, if
            % numVisibleEpochs==9, the user will only see 9 of the 10
            % labels, or if numVisibleEpochs==11, the user will only see 1
            % of the 10 labels.
            maxNumEpochsUntilNextRegime = iMaxPowerOfTenAtLeast10ThatIsAbove(numVisibleEpochs);
            epochPeriod = maxNumEpochsUntilNextRegime / 10;  % show a label on epochs that are a multiple of epochPeriod.
            epochIndices = epochPeriod * (1:10);
        end
    end
    
    methods(Access = private)
        function numVisibleEpochs = computeNumVisibleEpochs(this)
            % If the xbounds are such that the right-most epoch on the axes
            % is only partially showing, we count that epoch as visible
            % because any text label in it can be seen. 
            xLim = this.XLim;
            numVisibleEpochs = ceil( xLim(2) / this.EpochInfo.NumItersPerEpoch );
        end
    end
    
end

% helpers
function bounds = iComputeBoundsThatCoverSetOfBounds(cellOfBounds)
xMins = cellfun(@(bounds) iMinBound(bounds), cellOfBounds, 'UniformOutput', false);
xMaxs = cellfun(@(bounds) iMaxBound(bounds), cellOfBounds, 'UniformOutput', false);
minVal = min( [xMins{:}] );
maxVal = max( [xMaxs{:}] );
bounds = [minVal, maxVal];
end

function val = iMinBound(bounds)
if isempty(bounds)
    val = []; 
else
    val = bounds(1); 
end
end

function val = iMaxBound(bounds)
if isempty(bounds)
    val = []; 
else
    val = bounds(2); 
end
end

function bounds = iAddYPadding(bounds)
if ~isempty(bounds)
    if iRange(bounds) == 0
        padding = abs(0.05 * bounds(1));
    else
        padding = abs(0.05 * iRange(bounds));
    end
    bounds(2) = bounds(2) + padding;
end
end

function bounds = iUseDefaultBoundsIfEmpty(bounds)
if isempty(bounds)
   bounds = [0, 1]; 
end
end

function bounds = iAddPaddingToBoundsIfZeroRange(bounds)
if iRange(bounds) == 0
    midpoint = bounds(1);
    if midpoint == 0
        bounds = [0, 1];
    else
        padding = abs(midpoint) * 0.05;
        bounds = [midpoint - padding, midpoint + padding];
    end
end
end

function r = iRange(bounds)
assert( numel(bounds) == 2 );
r = abs(bounds(2) - bounds(1));
end

function powerOfTen = iMaxPowerOfTenAtLeast10ThatIsAbove(num)
power = ceil(log10(num));
powerOfTen = max(10, 10^power);
end

