classdef Stopwatch < nnet.internal.cnn.ui.adapter.Watch
    % Stopwatch   An adapter around tic-toc.
    
    %   Copyright 2017 The MathWorks, Inc.
    
    properties(Access = private)
        % TicHandle   (handle) The handle returned from tic
        TicHandle
    end
    
    methods
        function reset(this)
            this.TicHandle = tic;
        end
        
        function d = getDurationSinceReset(this)
            if isempty(this.TicHandle)
                d = NaN * seconds;
            else
                d = toc(this.TicHandle) * seconds;
            end
        end
    end
end
