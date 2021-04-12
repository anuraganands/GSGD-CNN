function varargout = executeWithStagedGPUOOMRecovery(computeFun, nOutputs, recoverFuns, layer) %#ok<INUSD>
% executeWithStagedGPUOOMRecovery   Generic utility to execute a function
% in a loop, catching GPU out-of-memory errors and making attempts to
% release GPU memory in a sequence.

%   Copyright 2017 The MathWorks, Inc.

nAttempts = numel(recoverFuns) + 1;
for attempt = 1:nAttempts
    try
        [ varargout{1:nOutputs} ] = computeFun();
        % Success - no need to loop
        break;
    catch me
        if attempt < nAttempts && ...
                me.identifier == "parallel:gpu:array:OOM"
            
            % Warn that we are incurring data transfer cost
            nnet.internal.cnn.util.gpuLowMemoryOneTimeWarning();
            
            % Uncomment this line to debug memory management
            %fprintf('Trying memory recovery strategy %d in layer %d\n', attempt, layer);
            recoverFuns{attempt}();
        else
            rethrow(me);
        end
    end
end
end
