classdef ValidationInfo < handle
    % ValidationInfo   Holds fixed information relating to validation
    
    %   Copyright 2017 The MathWorks, Inc.
    
    properties(SetAccess = private)
        % IsValidationEnabled   (logical) 
        IsValidationEnabled
        
        % ValidationFrequency   (double) 
        ValidationFrequency
        
        % ValidationPatience   (double)
        ValidationPatience
    end
    
    methods
        function this = ValidationInfo(isValidationEnabled, validationFrequency, validationPatience)
            this.IsValidationEnabled = isValidationEnabled;
            this.ValidationFrequency = validationFrequency;
            this.ValidationPatience = validationPatience;
        end
    end
    
end

