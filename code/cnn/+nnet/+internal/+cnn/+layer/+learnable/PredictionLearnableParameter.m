classdef PredictionLearnableParameter < nnet.internal.cnn.layer.learnable.LearnableParameter
    % PredictionLearnableParameter   Learnable parameter for use at prediction time
    %
    %   This class is used to represent a learnable parameter at prediction
    %   time, and it is slightly more complex than the representation used
    %   at training time.
    %
    %   This class can be used in GPU mode or host mode, and this is
    %   controlled by setting the property "UseGPU".
    %
    %   1) When "UseGPU" is false, the "Value" property will return the 
    %   host array stored in "PrivateValue".
    %   2) When "UseGPU" is true, the "Value" property will return a
    %   gpuArray stored in "CacheHandle".
    
    %   Copyright 2016-2017 The MathWorks, Inc.
    
    properties(Dependent)
        % Value   The value of the learnable parameter
        Value
    end
    
    properties(Access = private)        
        % CacheHandle   A handle class which holds a copy of the learnable parameter on the GPU
        CacheHandle
    end
    
    properties(SetAccess = private)
        % HostValue   The value of the learnable parameter
        HostValue
    end
    
    properties
        % UseGPU   A boolean value which determines of the GPU is used
        UseGPU
        
        % LearnRateFactor   Multiplier for the learning rate for this parameter
        %   A scalar double.
        LearnRateFactor

        % L2Factor   Multiplier for the L2 regularizer for this parameter
        %   A scalar double.
        L2Factor
    end
    
    methods
        function this = PredictionLearnableParameter()
            this.UseGPU = false;
            this.CacheHandle = nnet.internal.cnn.layer.learnable.CacheHandle();
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
        
        function val = getCacheHandle(this)
            val = this.CacheHandle;
        end

    end
    
    methods(Static)
        function obj = fromStruct(s)
            % Create from a structure, for use during loadobj
            obj = nnet.internal.cnn.layer.learnable.PredictionLearnableParameter();
            obj.Value = s.Value;
            obj.LearnRateFactor = s.LearnRateFactor;
            obj.L2Factor = s.L2Factor;
        end
    end

end
