function tf = isGPUCompatible()
% isGPUCompatible   Check if the currently selected GPU is compatible
%   tf = isGPUCompatible() will return true if the currently
%   selected GPU device can be used with the Convolutional Neural Network
%   feature, which requires an NVIDIA GPU with compute capability 3.0

%   Copyright 2016 The MathWorks, Inc.

if(iCanUsePCT() && parallel.gpu.GPUDevice.isAvailable())
    gpuInfo = gpuDevice();
    tf = iComputeCapabilityIsGreaterThanOrEqualToThree(gpuInfo);
else
    tf = false;
end
end

function tf = iComputeCapabilityIsGreaterThanOrEqualToThree(gpuInfo)
tf = str2double(gpuInfo.ComputeCapability) >= 3.0;
end

function tf = iCanUsePCT()
tf = nnet.internal.cnn.util.canUsePCT();
end