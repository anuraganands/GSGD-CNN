function gpuLowMemoryOneTimeWarning()
% gpuLowMemoryOneTimeWarning  Warn about GPU swapping with main memory
% and then disable.

% Copyright 2017, The MathWorks Inc.

warningId = 'nnet_cnn:warning:GPULowOnMemory';

% Warn without backtrace
warnState = warning('query', 'backtrace');
warning off backtrace;
warning( message(warningId) );
warning( warnState );

% Switch off the warning
warning( 'off', warningId );

end
