classdef NetworkInfo
    % NetworkInfo   Value class that stores computed information from SeriesNetwork
    
    %   Copyright 2017 The MathWorks, Inc.
    
    properties(SetAccess = private)
        % ShouldImageNormalizationBeComputed   (logical) Should image
        % normalization be computed for this network
        ShouldImageNormalizationBeComputed
    end
    
    methods
        function this = NetworkInfo(shouldImageNormalizationBeComputed)
            this.ShouldImageNormalizationBeComputed = shouldImageNormalizationBeComputed;
        end
    end
    
end

