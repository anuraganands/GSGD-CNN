classdef EpochAxesDisplayer < nnet.internal.cnn.ui.axes.EpochDisplayer
    % EpochAxesDisplayer   Helper class for computing the epoch rectangles and similar.
    
    %   Copyright 2017 The MathWorks, Inc.
    
    properties(Constant)      
        % XPaddingProportion   (double) X-Padding of an epoch label to avoid
        % epoch label being too close to the edge of the rectangle. The
        % padding is a proportion of the num iterations per epoch.
        XPaddingProportion = 0.025;
        
        % YPaddingProportion   (double) Y-Padding of an epoch label to avoid
        % epoch label being too close to the edge of the rectangle. The
        % padding is a proportion of the YBounds.
        YPaddingProportion = 0.02
    end
    
    methods
        function updateEpochRectangles(~, patchObj, numEpochs, numItersPerEpoch, yBounds)
            [low, high] = iIncreaseBounds(yBounds);
            
            [xData, yData] = iEpochRectangleData(numEpochs, numItersPerEpoch, low, high);
            patchObj.XData = xData;
            patchObj.YData = yData;
            patchObj.FaceColor = iGreyColor();
            patchObj.FaceAlpha = iAlpha();
            patchObj.EdgeColor = 'none';
        end
        
        function initializeEpochTexts(~, epochTextObjects)
            for i=1:numel(epochTextObjects)
                epochTextObjects(i).HorizontalAlignment = 'left';  % ensure horizontal position is relative to the left of the text.
                epochTextObjects(i).VerticalAlignment = 'bottom';  % ensure vertical position is relative to the bottom of the text.

                epochTextObjects(i).FontSize = 12;
                epochTextObjects(i).FontWeight = 'normal';
                
                epochTextObjects(i).Units = 'data';
                
                epochTextObjects(i).Color = iTextColor();
                
                epochTextObjects(i).Clipping = 'on';  % ensure that text gets clipped by the parent axes.
            end
        end
        
        function updateEpochTexts(this, epochTextObjects, epochIndices, numItersPerEpoch, xBounds, yBounds)
            assert(numel(epochTextObjects) == numel(epochIndices));
            
            % We compute the how many epochs are visible. This determines
            % the regime of how the texts are displayed.
            lastEpochIndexVisible = ceil(xBounds(2) / numItersPerEpoch);
            
            posX = iComputeXPosition(epochIndices, numItersPerEpoch, this.XPaddingProportion);
            posY = iComputeYPosition(yBounds, this.YPaddingProportion);
            
            for i=1:numel(epochTextObjects)
                % We only need to set positions and strings of the epoch
                % text objects which should be visible.
                if epochIndices(i) <= lastEpochIndexVisible
                    epochTextObjects(i).Position = [posX(i), posY, 0];           
                    iComputeAndSetEpochTextString(epochTextObjects(i), epochIndices(i), lastEpochIndexVisible);
                    epochTextObjects(i).Visible = 'on';
                else
                    epochTextObjects(i).Visible = 'off';
                end
            end
        end
    end
end

% helpers
function [low, high] = iIncreaseBounds(yBounds)
padding = 1000;
low  = min(yBounds(1), -10000) - padding;
high = max(yBounds(2),  10000) + padding;
end

function r = iRange(bounds)
r = abs(bounds(2) - bounds(1));
end

function [xData, yData] = iEpochRectangleData(numEpochs, numItersPerEpoch, low, high)
% iEpochRectangleData   All epoch rectangles plot vertices in the following
% order:
%    - lower left vertex   ( epochIndex,                  low  )
%    - upper left vertex   ( epochIndex,                  high )
%    - upper right vertex  ( epochIndex+numItersPerEpoch, high )
%    - lower right vertex  ( epochIndex+numItersPerEpoch, low  )

startIterIndices = iEvenEpochIndices(numEpochs) * numItersPerEpoch - (numItersPerEpoch - 1);  % draw rectangles on every even numbered epoch.
endIterIndices   = startIterIndices + numItersPerEpoch;   % rectangles end numItersPerEpoch after the start.
numRectangles = numel(startIterIndices);

xData = [startIterIndices; startIterIndices; endIterIndices; endIterIndices];
yData = repmat([low; high; high; low], 1, numRectangles);
end

function str = iEpochLabel(index)
str = message('nnet_cnn:internal:cnn:ui:trainingplot:EpochLabel', num2str(index)).getString();
end

function iComputeAndSetEpochTextString(epochTextObject, epochIndex, numVisibleEpochs) 
if iIsSmallEpochRegime(numVisibleEpochs)
    epochTextObject.String = iEpochLabel(epochIndex);
else
    epochTextObject.String = num2str(epochIndex, iIntegerFormat());
end
end

function xPos = iComputeXPosition(epochIndices, numItersPerEpoch, xPaddingProportion)
% if epochIndex==1, then position text at iteration 0, otherwise,
% if epochIndex==j, then position text at iteration numItersPerEpoch*(j-1)+1 
xPos = numItersPerEpoch * (epochIndices - 1) + 1; 
isEpochUnity = (epochIndices == 1);
xPos(isEpochUnity) = 0;
% add padding
xPadding = xPaddingProportion * numItersPerEpoch;
xPos = xPos + xPadding;
end

function yPos = iComputeYPosition(yBounds, yPaddingProportion)
% position text at 'fixed' position relative to the y-bounds.
yPadding = yPaddingProportion * iRange(yBounds);
yPos = yBounds(1) + yPadding;
end

function tf = iIsSmallEpochRegime(numVisibleEpochs)
tf = (numVisibleEpochs <= 10);
end

function vals = iEvenEpochIndices(numEpochs)
numEvenNumbers = floor(numEpochs/2);
vals = 2*(1:numEvenNumbers);
end

function color = iGreyColor()
color = [0.5, 0.5, 0.5];
end

function color = iTextColor()
color = [0.3, 0.3, 0.3];
end

function alpha = iAlpha()
alpha = 0.2;
end

function format = iIntegerFormat()
format = '%.0f';
end
