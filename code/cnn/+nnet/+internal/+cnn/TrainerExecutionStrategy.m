classdef TrainerExecutionStrategy
   % TrainerExecutionStrategy   Interface for Trainer execution strategies
   %
   %   A class that inherits from this interface will be used to implement
   %   calculations in the internal Trainer class on either the host or
   %   GPU.
   
   % Copyright 2016 The Mathworks, Inc.
   
    methods (Abstract)
       
        Y = environment(this, X)
        % Y = environment(this, X) is used by the Trainer to ensure that
        % data X used for training correctly associated with the hardware
        % used for training.
        
        [accumI, numImages] = computeAccumImage(this, distributedData, augmentations );
        % [accumI, numImages] = computeAccumImage(this, distributedData,
        % augmentations ) computes the accumulated image of the training
        % data, which is used during the average image calculation.
        
    end
    
    methods (Access = public)
        
        function avgI = computeAverageImage(this, data, augmentations, executionSettings)
            % Average image is computed in parallel or in serial
            if executionSettings.useParallel
                avgI = this.computeAverageImageParallel(data, augmentations);
            else
                avgI = this.computeAverageImageSerial(data, augmentations);
            end
            avgI = gather( avgI );
        end
       
        function avgI = computeAverageImageSerial(this, data, augmentations)
            [accumI, numImages] = this.computeAccumImage(data, augmentations);
            avgI = accumI ./ numImages;
        end
        
        function avgI = computeAverageImageParallel(this, data, augmentations)
            % Distribute average image computation to the pool
            [avgI, location] = data.computeInParallel( @this.computeAverageImageRemote, 1, data.DistributedData, augmentations );
            avgI = avgI{location};
            avgI = avgI{1}; % First output from my remote function
        end
                
    end
    
    methods (Access = private)
        
        function avgI = computeAverageImageRemote( this, data, augmentations )
            % Compute average on each worker
            [accumI, numImages] = this.computeAccumImage( data, augmentations );
            % Compute combined average
            accumI = gplus(accumI, 1, class(accumI));
            numImages = gplus(numImages, 1);
            if labindex == 1
                avgI = accumI ./ numImages;
            else
                avgI = [];
            end
            % Always output on the host in case client has no GPU
            avgI = gather(avgI);
        end
           
    end
    
end