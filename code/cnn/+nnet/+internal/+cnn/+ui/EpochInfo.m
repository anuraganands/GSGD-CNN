classdef EpochInfo < handle
    % EpochInfo   Class storing the information relating to epochs
    
    %   Copyright 2017 The MathWorks, Inc.
    
    properties(SetAccess = private)
        % NumEpochs   (integer) The total number of epochs that it's
        % possible to go through during training
        NumEpochs
        
        % NumItersPerEpoch   (integer) The number of iterations per epoch
        NumItersPerEpoch
        
        % NumIters   (integer) The total number of iterations
        NumIters
    end
    
    methods
        function this = EpochInfo(numEpochs, numObservations, miniBatchSize)
            this.NumEpochs = numEpochs;
            this.NumItersPerEpoch = max(1, floor(numObservations / miniBatchSize));
            this.NumIters = this.NumEpochs * this.NumItersPerEpoch; 
        end
    end
    
end

