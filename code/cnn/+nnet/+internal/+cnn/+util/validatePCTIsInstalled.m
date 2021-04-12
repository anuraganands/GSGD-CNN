function validatePCTIsInstalled(executionEnvironment)
% validatePCTIsInstalled   Throw an error if PCT is not installed

%   Copyright 2016 The MathWorks, Inc.

if(~nnet.internal.cnn.util.canUsePCT())
    error( message( ...
        'nnet_cnn:internal:cnn:util:validatePCTIsInstalled:PCTNotInstalled', ...
        executionEnvironment) );
end

end