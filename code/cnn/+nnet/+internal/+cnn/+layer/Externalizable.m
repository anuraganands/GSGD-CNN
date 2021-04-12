classdef(Abstract) Externalizable
    % Externalizable   Mixin to make built-in network layers externalizable
    
    %   Copyright 2017 The MathWorks, Inc.
    
    properties(Access = {?nnet.internal.cnn.layer.Externalizable, ?nnet.internal.cnn.layer.util.ExternalInternalConverter})
        % PrivateLayer (nnet.internal.cnn.layer.Layer)   An internal
        % built-in layer
        PrivateLayer
    end
    
    methods (Sealed)
        function aLayer = setLearnRateFactor( aLayer, learnableParameterName, learnRateFactor )
            iAssertValidFactor( learnRateFactor );
            try
                aLayer.PrivateLayer.(learnableParameterName).LearnRateFactor = learnRateFactor;
            catch
                iThrowParameterNotFound( learnableParameterName );
            end
        end
        
        function learnRateFactor = getLearnRateFactor( aLayer, learnableParameterName )
            try
                learnRateFactor = aLayer.PrivateLayer.(learnableParameterName).LearnRateFactor;
            catch
                iThrowParameterNotFound( learnableParameterName );
            end
        end
        
        function aLayer = setL2Factor( aLayer, learnableParameterName, l2Factor )
            iAssertValidFactor( l2Factor );
            try
                aLayer.PrivateLayer.(learnableParameterName).L2Factor = l2Factor;
            catch
                iThrowParameterNotFound( learnableParameterName );
            end
        end
        
        function l2Factor = getL2Factor( aLayer, learnableParameterName )
            try
                l2Factor = aLayer.PrivateLayer.(learnableParameterName).L2Factor;
            catch
                iThrowParameterNotFound( learnableParameterName );
            end
        end
    end
end

function iAssertValidFactor(value)
try
    nnet.internal.cnn.layer.paramvalidation.validateLearnFactor(value);
catch e
    % Reduce the stack trace of the error message by throwing as caller
    throwAsCaller(e)
end
end

function iThrowParameterNotFound( learnableParameterName )
error( message( 'nnet_cnn:internal:cnn:layer:Externalizable:ParameterNameNotFound', learnableParameterName ) )
end