classdef CachedParameter
    % CachedParameter   Non-learnable parameter with optional GPU cache
    %
    %   This class can be used in GPU mode or host mode, and this is
    %   controlled by setting the property "UseGPU".
    %
    %   1) When "UseGPU" is false, the "Value" property will return the
    %   host array stored in "HostValue".
    %   2) When "UseGPU" is true, the "Value" property will return a
    %   gpuArray stored in "CacheHandle".
    %
    %   The HostValue property always returns the HostValue.
    
    %   Copyright 2017 The MathWorks, Inc.
    
    properties
        % UseGPU   A boolean value which determines if the GPU is used
        UseGPU
    end
    
    properties(Dependent)
        % Value   The value of the learnable parameter
        Value
    end
    
    properties(SetAccess = private)
        % HostValue   The value of the learnable parameter
        HostValue
    end
    
    properties(Access = private)
        % CacheHandle   A handle class which holds a copy of the parameter on the GPU
        CacheHandle
    end
    
    methods
        function this = CachedParameter(val)
            this.UseGPU = false;
            this.CacheHandle = nnet.internal.cnn.layer.learnable.CacheHandle();
            if nargin
                this.Value = val;
            end
        end
        
        function this = set.Value(this, val)
            this.HostValue = gather(val); % Just in case we are given a gpuArray!
            
            % Update the cache based on a new HostValue. We always switch
            % to a new cache if the value is changed to avoid inadvertantly
            % linking two copies of a parameter.
            if this.UseGPU
                gpuValue = parallel.internal.gpu.CachedGPUArray(this.HostValue);
                this.CacheHandle = nnet.internal.cnn.layer.learnable.CacheHandle(gpuValue);
            else
                % If not using the GPU, just create a new empty cache
                this.CacheHandle = nnet.internal.cnn.layer.learnable.CacheHandle();
            end
        end
        
        function val = get.Value(this)
            if this.UseGPU
                % Make sure the cache is filled, and get hold of the
                % contents. Note that we used a CachedGPUArray which is a
                % special GPUArray that is robust to the device being
                % reset.
                if this.CacheHandle.isEmpty()
                    gpuCache = parallel.internal.gpu.CachedGPUArray(this.HostValue);
                    this.CacheHandle.fillCache(gpuCache);
                else
                    gpuCache = this.CacheHandle.Value;
                end
                val = gpuCache.GPUValue;
                
            else
                val = this.HostValue;
                
            end
        end
        
    end
    
end
