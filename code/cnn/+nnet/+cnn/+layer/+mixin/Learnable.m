classdef (Abstract) Learnable < nnet.internal.cnnhost.LearnableAttributeClass
    % Learnable     Learnable mixin for convolutional neural network layers
    %   This class defines a "Learnable" attribute tag, together with
    %   setters and getters for learn rate and L2 factors of the learnable
    %   parameters
    
    %   Copyright 2017 The MathWorks, Inc.
    
    properties (Access = private)    
        % LearnRateFactor (struct)   Struct holding a map between learnable
        % parameter names and their learn rate factors
        LearnRateFactor
        
        % L2Factor (struct)   Struct holding a map between learnable
        % parameter names and their L2 factors
        L2Factor    
    end    
    
    properties (Access = private, Constant)
        % DefaultLearnRateFactor   Default learn rate factor
        DefaultLearnRateFactor = 1;
        
        % DefaultL2Factor   Default L2 factor
        DefaultL2Factor = 1;
    end
    
    methods
        function aLayer = Learnable()
            aLayer = discoverLearnableParameters( aLayer );
        end
    end
    
    methods (Sealed)
        function aLayer = setLearnRateFactor( aLayer, learnableParameterName, learnRateFactor )
            assertLearnableParameterExists( aLayer,learnableParameterName );
            iAssertValidFactor( learnRateFactor );
            aLayer.LearnRateFactor.(learnableParameterName) = learnRateFactor;
        end
        
        function learnRateFactor = getLearnRateFactor( aLayer, learnableParameterName )
            assertLearnableParameterExists(aLayer,learnableParameterName);
            learnRateFactor = aLayer.LearnRateFactor.(learnableParameterName);
        end
        
        function aLayer = setL2Factor( aLayer, learnableParameterName, l2Factor )
            assertLearnableParameterExists(aLayer,learnableParameterName);
            iAssertValidFactor( l2Factor );
            aLayer.L2Factor.(learnableParameterName) = l2Factor;
        end
        
        function l2Factor = getL2Factor( aLayer, learnableParameterName )
            assertLearnableParameterExists(aLayer,learnableParameterName);
            l2Factor = aLayer.L2Factor.(learnableParameterName);
        end
    end
    
    methods (Access = private)
        function assertLearnableParameterExists( aLayer, learnableParameterName )
            % assertLearnableParameterExists   Error out if
            % learnableParameterName is not a learnable parameter of aLayer
            if ~isALearnableParameter( aLayer, learnableParameterName )
                error( message( 'nnet_cnn:layer:mixin:Learnable:ParameterNameNotFound', learnableParameterName ) )
            end
        end
        
        function tf = isALearnableParameter( aLayer, learnableParameterName )
            % isALearnableParameter   To check if a learnableParameterName
            % corresponds to an existing learnable parameter, we simply
            % check if the name is a field of the L2Factor struct that has
            % been pre-filled at construction time
            tf = isfield( aLayer.L2Factor, learnableParameterName );
        end
        
        function aLayer = discoverLearnableParameters( aLayer )
            % discoverLearnableParameters   Discover learnable parameters
            % and add them to the learn rate and L2 factor structures with
            % initial values set to 1. This way we don't need to access the
            % meta class each time we want to check if a learnable
            % parameter exists
            
            propertyList = iPropertyList(aLayer);
            for ii=1:numel(propertyList)
                if iIsLearnable( propertyList(ii) )
                    % Set initial learn rate factor and L2 factor
                    name = propertyList(ii).Name;
                    aLayer.LearnRateFactor.(name) = aLayer.DefaultLearnRateFactor;
                    aLayer.L2Factor.(name) = aLayer.DefaultL2Factor;
                end
            end
        end
    end
end

function propertyList = iPropertyList( layer )
metaLayer = metaclass(layer);
propertyList = metaLayer.PropertyList;
end

function tf = iIsLearnable( prop )
tf = isprop(prop, 'Learnable') && prop.Learnable;
end

function iAssertValidFactor(value)
try
    nnet.internal.cnn.layer.paramvalidation.validateLearnFactor(value);
catch e
    % Reduce the stack trace of the error message by throwing as caller
    throwAsCaller(e)
end
end