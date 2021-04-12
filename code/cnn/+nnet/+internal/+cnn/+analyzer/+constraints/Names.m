classdef Names < nnet.internal.cnn.analyzer.constraints.Constraint
    % Names         Constrain object to be used by analyzeNetwork.
    %               Detects issues related to duplicated layer names in
    %               the network.
    
    %   Copyright 2017 The MathWorks, Inc.
    
    methods
        
        function testDuplicatedNames(test)
            % Test whether a layer had duplicated names and required
            % renaming some of the layers.
            
            original = string({test.LayerAnalyzers.OriginalName});
            names = string({test.LayerAnalyzers.Name});
            
            renamed = ( names ~= original );
            deduced = ( original == "" );
            
            duplicated = ( renamed & ~deduced );
            
            for i=find(duplicated)
                test.addLayerWarning(i, "Names:DuplicatedNames", ...
                    names(i), original(i));
            end
        end
        
    end
end