classdef MetaLayer
    % MetaLayer   Class that holds meta information about a
    % nnet.layer.Layer object. To check the validity of a layer, use
    % checkLayer
    
    %   Copyright 2017 The MathWorks, Inc.
    
    properties (SetAccess = private)
        % Predict (meta.method)   Meta method containing information about
        % the predict method
        Predict
        
        % Forward (meta.method)   Meta method containing information about
        % the forward method
        Forward
        
        % Backward (meta.method)   Meta method containing information about
        % the backward method
        Backward
        
        % ParametersNames (cell array of chars)   Names of the learnable
        % parameters
        ParametersNames

        % Parameters (cell array)   Values of the learnable parameters
        ParametersValues
        
        % IsForwardDefined (logical)   Flag set to true if 'forward' is
        % overridden in the layer's class definition
        IsForwardDefined
    end
    
    methods
        function this = MetaLayer( aLayer )
            [this.Predict, this.Forward, this.Backward] = iGetMethods( aLayer );
            [this.ParametersNames, this.ParametersValues] = iDiscoverLearnableParameters( aLayer );
            this.IsForwardDefined = iIsForwardDefined( aLayer );
        end
    end
end

function [predict, forward, backward] = iGetMethods( layer )
% iGetMethods   Get predict, forward and backward meta methods from the
% layer
methodList = iMethodList( layer );
for ii=1:numel(methodList)
    currentMethod = methodList(ii);
    switch currentMethod.Name
        case 'predict'
            predict = currentMethod;
        case 'forward'
            forward = currentMethod;
        case 'backward'
            backward = currentMethod;
        otherwise
    end
end
end

function [names, values] = iDiscoverLearnableParameters( layer )
% iDiscoverLearnableParameters   Discover learnable parameters declared in
% the layer

names = {};
values = {};

propertyList = iPropertyList(layer);
for ii=1:numel(propertyList)
    if isprop(propertyList(ii), 'Learnable') && propertyList(ii).Learnable
        name = propertyList(ii).Name;
        % Store the parameter name
        names{end+1} = name; %#ok<AGROW>
        % Create and store matching learnable parameter value
        currentValue = layer.(name);
        values{end+1} = currentValue; %#ok<AGROW>        
    end
end
end

function propertyList = iPropertyList( layer )
metaLayer = metaclass(layer);
propertyList = metaLayer.PropertyList;
end

function functionList = iMethodList( layer )
metaLayer = metaclass(layer);
functionList = metaLayer.MethodList;
end

function tf = iIsForwardDefined( layer )
% iIsForwardDefined   Return true if forward was overriden in the layer's
% class definition
tf = false;
functionList = iMethodList( layer );
for ii=1:numel(functionList)
    currentFunction = functionList(ii);
    if isequal( currentFunction.Name, 'forward' )
        definingClass = currentFunction.DefiningClass.Name;
        % If the defining class of 'forward' was the Layer super-class,
        % then the function has not been overridden by the user
        tf = ~isequal( definingClass, 'nnet.layer.Layer' );
    end
end
end
