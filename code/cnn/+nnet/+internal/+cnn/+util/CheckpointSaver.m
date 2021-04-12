classdef CheckpointSaver < nnet.internal.cnn.util.Reporter
    % CheckpointSaver   Reporter to save trained network cheptoints
    
    %   Copyright 2015-2017 The MathWorks, Inc.
    
    properties
        CheckpointPath
        
        % ConvertorFcn   Handle to function to convert an internal network
        % before saving it
        %
        % Default: identity, i.e., don't convert
        ConvertorFcn = @(x)x
    end
    
    methods
        function this = CheckpointSaver( path )
            this.CheckpointPath = path;
        end
        
        function setup( ~ )
        end
        
        function start( ~ )
        end
        
        function reportIteration( ~, ~ )
        end
        
        function reportEpoch( this, ~, iteration, network )
            this.saveCheckpoint( network, iteration );
        end
        
        function finish( ~, ~ )
        end
    end
    
    methods(Access = private)
        function saveCheckpoint(this, net, iteration)
            checkpointPath = this.CheckpointPath;
            name = iGenerateCheckpointName(iteration);
            fullPath = fullfile(checkpointPath, name);

            network = this.ConvertorFcn(net);
            
            iSaveNetwork(fullPath, network);
        end
    end
end

function iSaveNetwork(fullPath, network)
try
    iSave(fullPath, 'net', network);
catch e
    warning( message('nnet_cnn:internal:cnn:Trainer:SaveCheckpointFailed', fullPath, e.message ) );
end
end

function iSave(fullPath, name, value) 
S.(name) = value; %#ok<STRNU>
save(fullPath, '-struct', 'S', name);
end


function name = iGenerateCheckpointName(iteration)
basename = 'convnet_checkpoint';
timestamp = char(datetime('now', 'Format', 'yyyy_MM_dd__HH_mm_ss'));
name = [ basename '__' int2str(iteration) '__' timestamp '.mat' ];
end
