classdef Colors
    % Colors   Provides commonly used colors

    %   Copyright 2017 The MathWorks, Inc.
    
    properties(Constant)
        % Regression
        RegressionRMSELineColor = [0, 0.4470, 0.7410];   % color1
        RegressionRMSEDotColor = [178, 212, 235]/255;
        RegressionValidationRMSELineColor = [64 64 64]/255;
        
        RegressionLossLineColor = [0.8500, 0.3250, 0.0980];  % color2
        RegressionLossDotColor = [244, 203, 186]/255;
        RegressionValidationLossLineColor = [64 64 64]/255;
        
        % Classification
        ClassificationAccuracyLineColor = [0, 0.4470, 0.7410];   % color1
        ClassificationAccuracyDotColor = [178, 212, 235]/255;
        ClassificationValidationAccuracyLineColor = [64 64 64]/255;
        
        ClassificationLossLineColor = [0.8500, 0.3250, 0.0980];  % color2
        ClassificationLossDotColor = [244, 203, 186]/255;
        ClassificationValidationLossLineColor = [64 64 64]/255;

        % Hyperlinks
        Hyperlink = [0, 102, 204]/256;
        SelectedHyperlink = [100, 102, 204]/256;
        VisitedHyperlink = [0, 51, 102]/256;
    end
end
