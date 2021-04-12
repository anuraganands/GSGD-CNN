function tf = GPUShouldBeUsed(executionEnvironment, workerLoad)
% GPUShouldBeUsed   This utility function determines whether a GPU is
% available for calculation.

%   Copyright 2016 The MathWorks, Inc.

switch executionEnvironment
    case 'cpu'
        tf = false;
    case 'gpu'
        iValidatePCTIsInstalled(executionEnvironment);
        tf = nnet.internal.cnn.util.isGPUCompatible();
        if ~tf
            error(message('nnet_cnn:internal:cnngpu:GPUArchMismatch'));
        end
    case 'auto'
        tf = nnet.internal.cnn.util.isGPUCompatible();
    case {'multi-gpu', 'parallel'}
        iValidatePCTIsInstalled(executionEnvironment);
        poolIsOpen = ~isempty( gcp('nocreate') );
        if poolIsOpen
            spmd
                % Ignore disabled workers when deciding whether others can be used
                tf = gop(@and, ...
                    workerLoad(labindex)==0 || nnet.internal.cnn.util.isGPUCompatible(), ...
                    1);
                tf = feval('distributedutil.AutoTransfer', tf, 1 );
            end
            tf = tf.Value;
        else
            tf = nnet.internal.cnn.util.isGPUCompatible();
        end
        if executionEnvironment == "multi-gpu" && ~tf
            error(message('nnet_cnn:internal:cnngpu:GPUArchMismatch'));
        end
    otherwise
        error(message('nnet_cnn:SeriesNetwork:InvalidExecutionEnvironment'));
end
end

function iValidatePCTIsInstalled(executionEnvironment)
nnet.internal.cnn.util.validatePCTIsInstalled(executionEnvironment);
end
