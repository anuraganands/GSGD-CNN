classdef(Abstract) Watch < handle
    % Watch   An interface for timing
    
    %   Copyright 2017 The MathWorks, Inc.
    
    methods(Abstract)
        reset(this)
        % reset   Reset the watch
        
        d = getDurationSinceReset(this)
        % getDurationSinceReset   Get the duration between this call and
        % when reset was last called. This should error if this is called
        % without reset being called first.
    end
end
