classdef EpochDisplayHider < nnet.internal.cnn.ui.axes.EpochDisplayer
    % EpochDisplayHider   Class which hides any epoch display
    
    %   Copyright 2017 The MathWorks, Inc.
    
    methods
        function updateEpochRectangles(~, patchObj, ~, ~, ~)
            patchObj.Visible = 'off';
        end
        
        function initializeEpochTexts(~, epochTextObjects)
            for i=1:numel(epochTextObjects)
                epochTextObjects(i).Visible = 'off'; 
            end
        end
        
        function updateEpochTexts(~,~,~,~,~,~)
        end
    end
end

