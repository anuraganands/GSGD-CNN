% Summary for deepDreamImage
%
% Copyright 2016 The MathWorks, Inc.
classdef DeepDreamImageSummary < handle    
    
    properties
        % Octave Current octave being processed.
        Octave
        
        % Iteration Current iteration being processed.
        Iteration
        
        % Activation Current activation strength.
        ActivationStrength
    end
    
    
    methods
        function update(this, octave, iteration, activations)
           this.Iteration = iteration; 
           
           this.Octave = octave;
                      
           this.ActivationStrength = iChannelActivationNorm(activations);
        end
    end

end

function m = iChannelActivationNorm(activations)
% Report the activation summary as the L2 norm of channel activations. This
% is a simple way to report how the optimization of multiple channels is
% progressing. 

activations = squeeze(activations);

m = norm(activations,2);
end