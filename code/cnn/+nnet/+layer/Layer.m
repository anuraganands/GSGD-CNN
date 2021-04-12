classdef (Abstract) Layer < nnet.cnn.layer.Layer & nnet.cnn.layer.mixin.Learnable
    % Layer   Interface for custom network layers
    %
    %   Copyright 2017 The MathWorks, Inc.
    
    properties
        % Name (char vector)   A name for the layer
        Name = '';
    end
    
    properties (SetAccess = protected)
        % Description (char vector)   A one line description for the layer
        Description
        
        % Type (char vector)   The type of layer
        Type
    end
    
    methods (Abstract)
        % predict   Forward input data through the layer and output the result
        %
        % Syntax
        %   Z = predict( aLayer, X )
        %
        % Inputs
        %   aLayer - the layer to forward through
        %   X - the input to forward propagate through the layer
        %
        % Output
        %   Z - the output of forward propagation through the layer
        Z = predict( aLayer, X )
        
        % backward    Back propagate the derivative of the loss function
        % through one layer
        %
        % For notation, assume that a layer is represented by the function
        % Z = f(X,W) where X is the input data to the layer, W the
        % learnable weights, and Z the layer output. The derivatives of the
        % loss function with respect to X, W, and Z are denoted dLdX, dLdW,
        % and dLdZ.
        %
        % Syntax
        %   [dLdX, dLdW1, ..., dLdWn] = backward( aLayer, X, Z, dLdZ, memory )
        %
        % Inputs
        %   aLayer - the layer to backprop through
        %   X      - the input that was used for forward propagation
        %            through the layer
        %   Z      - the output from forward propagation through the layer
        %   dLdZ   - the derivative of the loss (L) with respect to the
        %            output (Z). This is usually obtained via
        %            back-propagation from the next layer in the network
        %   memory - whatever "memory" that was produced by forward
        %            propagation through the layer using the forward method
        %
        % Outputs
        %   dLdX              - the derivative of the loss function with
        %                       respect to X. It must have the same
        %                       dimension as X
        %   dLdW1, ..., dLdWn - the derivatives of the loss (L) with
        %                       respect to the n learnable parameters (W1,
        %                       ..., Wn). Each derivative must have the
        %                       same dimension as the parameter it refers to
        %
        % See also: forward
        [dLdX, varargout] = backward( aLayer, X, Z, dLdZ, memory )
    end
    
    methods
        % forward   Forward input data through the layer and output the
        % result. If forward is not overridden, Z will be the output of
        % predict and memory will be an empty numeric array
        %
        % Syntax
        %   [Z, memory] = forward( aLayer, X )
        %
        % Inputs
        %   aLayer - the layer to forward through
        %   X - the input to forward propagate through the layer
        %
        % Outputs
        %   Z - the output of forward propagation through the layer
        %   memory - "memory" produced by forward propagation through the layer
        function [Z, memory] = forward( aLayer, X )
            Z = predict( aLayer, X );
            memory = [];
        end
        
    end
    
    methods(Hidden, Access = protected)
        function layer = Layer()
            try
                className = matlab.mixin.CustomDisplay.getClassNameForHeader( layer );
                nnet.internal.cnn.layer.util.CustomLayerVerifier...
                    .validateMethodSignatures(layer, className);
            catch e
                % Wrap exception in a CNNException, which reports the error in a custom way
                err = nnet.internal.cnn.util.CNNException.hBuildCustomError( e );
                throwAsCaller(err);
            end
        end
        
        function [description, type] = getOneLineDisplay( layer )
            if isempty( layer.Description )
                % If Description was not defined, show the class
                description = class( layer );
            else
                description = layer.Description;
            end
            
            if isempty( layer.Type )
                % If Type was not defined, show the class
                type = class( layer );
            else
                type = layer.Type;
            end
        end
        
        function groups = getPropertyGroups( this )
            [ generalProperties, learnableParameters ] = iGetLayerProperties( this );
            
            groups = [
                this.propertyGroupGeneral( generalProperties )
                this.propertyGroupLearnableParameters( learnableParameters )
                ];
        end
        
        function footer = getFooter( this )
            variableName = inputname(1);
            footer = this.createShowAllPropertiesFooter( variableName );
        end
    end
    
    methods (Hidden, Sealed)
        function displayAllProperties(this)
            proplist = properties( this );
            proplist = iOrderPropList( proplist );
            matlab.mixin.CustomDisplay.displayPropertyGroups( ...
                this, ...
                this.propertyGroupGeneral( proplist ) );
        end
    end
    
    methods
        function layer = set.Name( layer, val )
            iAssertValidLayerName( val );
            layer.Name = char( val );
        end
    end
end

function messageString = iGetMessageString( messageID )
messageString = getString( message( messageID ) );
end

function iAssertValidLayerName( name )
nnet.internal.cnn.layer.paramvalidation.validateLayerName( name );
end

function [ generalProperties, learnableParameters ] = iGetLayerProperties( aLayer )
% iGetLayerProperties   Return layer's properties divided into general
% properties and learnable parameters
learnableParameters = {};
generalProperties = {};
propertyList = iPropertyList(aLayer);
for ii=1:numel(propertyList)
    if iIsLearnable( propertyList(ii) )
        learnableParameters{end+1} = propertyList(ii).Name; %#ok<AGROW>
    elseif iCanBeShown( propertyList(ii) )
        generalProperties{end+1} = propertyList(ii).Name; %#ok<AGROW>
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

function tf = iCanBeShown( prop )
% iCanBeShown   Return true if the property can be shown. Properties that
% cannot be shown are Hidden or they belong to a list of properties that
% are not supposed to be shown in the short display (e.g. Type,
% Description)
notShowList = {'Description', 'Type'};
tf = ~prop.Hidden && ~ismember( prop.Name, notShowList );
end

function p = iOrderPropList( p )
% iOrderPropList   Reorder property list p to make sure Name, Description
% and Type are reported first
idx = contains(p, {'Name', 'Description', 'Type'});
p = [p(idx); p(~idx)];
end