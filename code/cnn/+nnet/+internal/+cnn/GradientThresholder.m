classdef GradientThresholder
    % GradientThresholder   Class for creating gradient thresholders
    %
    %   This class should be used to create norm based and value based
    %   gradient thresholders.
    %
    %   GradientThresholder properties:
    %       Method                      - Method for gradient thresholding.
    %       Threshold                   - Gradient threshold.
    %
    %   GradientThresholder methods:
    %       GradientThresholder         - Create a gradient thresholder by
    %                                     providing gradient thresholding
    %                                     options in the form of a
    %                                     structure.
    %       thresholdGradients          - Perform the thresholding based on
    %                                     the Method and Threshold.
    
    %   Copyright 2017 The MathWorks, Inc.
    
    properties(SetAccess = private, GetAccess = public)
        Method
        Threshold
    end
    
    methods
        function self = GradientThresholder(gradientThresholdOptions)
            % GradientThresholder   Create a gradient thresholder
            %
            % Input gradientThresholdOptions is a struct with following
            % fields:
            %   Method      - global-l2norm', 'l2norm'(default), 
            %                 'absolute-value'
            %   Threshold   - Positive scalar. Default is Inf.
            
            if nargin > 0
                self.Method = gradientThresholdOptions.Method;
                self.Threshold = gradientThresholdOptions.Threshold;
            else
                self.Method = 'l2norm';
                self.Threshold = Inf;
            end
        end
    end
    
    methods(Hidden)
        function originalGrads = thresholdGradients(self, originalGrads)
            % thresholdGradients    Performs gradient thresholding
            %
            % Input originalGrads is a cell array of gradients for all
            % learnable parameters.
            
            if isinf(self.Threshold)
                return;
            else
                validGradsMask = ~cellfun(@isempty, originalGrads);
                validGrads = originalGrads(validGradsMask);
                if strcmp(self.Method,'l2norm')
                    validGrads = iThresholdL2Norm(validGrads, self.Threshold);
                elseif strcmp(self.Method,'global-l2norm')
                    validGrads = iThresholdGlobalL2Norm(validGrads, self.Threshold);
                elseif strcmp(self.Method,'absolute-value')
                    validGrads = iThresholdAbsoluteValue(validGrads, self.Threshold);
                end
                originalGrads(validGradsMask) = validGrads;
            end
        end
    end
end

function originalGrads = iThresholdL2Norm(originalGrads, normThreshold)
for i = 1:numel(originalGrads)
    squareSum = sum(originalGrads{i}(:).^2);
    gradNorm = sqrt(squareSum);
    if gradNorm > normThreshold
        originalGrads{i} = originalGrads{i}*(normThreshold/gradNorm);
    end
end
end

function originalGrads = iThresholdGlobalL2Norm(originalGrads, normThreshold)
globalL2Norm = 0;
for i = 1:numel(originalGrads)
    globalL2Norm = globalL2Norm + sum(originalGrads{i}(:).^2);
end
globalL2Norm = sqrt(globalL2Norm);
if globalL2Norm > normThreshold
    normScale = (normThreshold/globalL2Norm);
    for i = 1:numel(originalGrads)
        originalGrads{i} = originalGrads{i}*normScale;
    end
end
end

function originalGrads = iThresholdAbsoluteValue(originalGrads, absValueThreshold)
for i = 1:numel(originalGrads)
    originalGrads{i}(originalGrads{i}>absValueThreshold) = absValueThreshold;
    originalGrads{i}(originalGrads{i}<-absValueThreshold) = -absValueThreshold;
end
end