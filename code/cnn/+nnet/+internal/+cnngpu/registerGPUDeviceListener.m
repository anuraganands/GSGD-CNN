function registerGPUDeviceListener
% nnet.internal.cnngpu.registerGPUDeviceListener
%
% This function adds a global listener to the DeviceDeselecting
% event in the parallel.gpu.GPUDeviceManager, calling the cnngpu
% library's reset function to ensure any persistent state is
% cleaned up when the GPU device becomes invalid.

% Copyright 2015 The MathWorks, Inc.

addlistener(parallel.gpu.GPUDeviceManager.instance(), ...
    'DeviceDeselecting', @nnet.internal.cnngpu.reset);

end
