classdef StopReason 
    % StopReason   Enumerates the different reasons why training stopped
    
    %   Copyright 2017 The MathWorks, Inc.
    
    enumeration
        % FinalIteration
        FinalIteration
        
        % StopButton
        StopButton
        
        % ValidationStopping
        ValidationStopping
        
        % OutputFcn
        OutputFcn
    end
end

