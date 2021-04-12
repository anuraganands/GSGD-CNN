classdef LayerProxy < nnet.cnn.layer.Layer
    % LayerProxy        This class is intended to be used by the class
    %                   LayerAnalyzer as a proxy to access protected
    %                   methods of the external layers.
    %
    %   LayerProxy properties:
    %       Name            Read access to the layer name
    %       Type            Read access to the type of layer message
    %       Description     Read access to the description of layer message
    %
    %       Properties      String vector with the name of the (standard)
    %                       properties of the layer.
    %       HyperParameters, LearnableParameters, DynamicParameters
    %                       String vectors with the name of the
    %                       parameters of the layer.
    %       
    %       InputPorts      String vectors with the name of the input ports
    %                       of the layer.
    %       OutputPorts     String vectors with the name of the output
    %                       ports of the layer.
    %
    %       ExternalLayer   The external layer used to create the
    %                       LayerProxy object.
    %       InternalLayer   The external layer associated with the
    %                       ExternalLayer.
    
    %   Copyright 2017 The MathWorks, Inc.
    
    properties (Access = private)
        
        TargetLayer
        
    end
        
    properties (Dependent)
        
        % We need to define Name because it is abstract in
        % nnet.cnn.layer.Layer
        Name
        
        Type
        Description
        
        Properties
        
        HyperParameters
        DynamicParameters
        LearnableParameters
        
        InputPorts
        OutputPorts
        
        ExternalLayer
        InternalLayer
        
    end
    
    methods
        
        function this = LayerProxy(TargetLayer)
            this.TargetLayer = TargetLayer;
        end
        
        function v = get.Name(this)
            v = this.TargetLayer.Name;
        end
        
        function v = get.Type(this)
            [~, v] = this.TargetLayer.getOneLineDisplay();
        end
        function v = get.Description(this)
            [v, ~] = this.TargetLayer.getOneLineDisplay();
        end
        
        function p = get.Properties(this)
            p = this.getPropertyNameFromGroup("StandadProperties");
            p(p == "Name") = [];
        end
        
        function p = get.HyperParameters(this)
            p = this.getPropertyNameFromGroup("HyperParameters");
        end
        
        function p = get.DynamicParameters(this)
            p = this.getPropertyNameFromGroup("DynamicParameters");
        end
        
        function p = get.LearnableParameters(this)
            p = this.getPropertyNameFromGroup("LearnableParameters");
        end
        
        function names = get.InputPorts(this)
            layer = this.ExternalLayer;
            internalLayer = this.InternalLayer;
            
            if isa(layer, 'nnet.cnn.layer.MaxUnpooling2DLayer')
                names = ["in"; "indices"; "size"];
            elseif isa(layer, 'nnet.cnn.layer.AdditionLayer')
                n = layer.NumInputs;
                names = "in" + (1:n)';
            elseif isa(layer, 'nnet.cnn.layer.DepthConcatenationLayer')
                n = layer.NumInputs;
                names = "in" + (1:n)';
            elseif isa(layer, 'nnet.cnn.layer.Crop2DLayer')
                names = ["in"; "ref"];
            elseif isa(internalLayer, 'nnet.internal.cnn.layer.ImageInput')
                names = string.empty;
            elseif isa(internalLayer, 'nnet.internal.cnn.layer.SequenceInput')
                names = string.empty;
            else
                names = "in";
            end
        end
        
        function names = get.OutputPorts(this)
            layer = this.TargetLayer;
            internalLayer = this.InternalLayer;
            
            if isa(layer, 'nnet.cnn.layer.MaxPooling2DLayer') ...
               && layer.HasUnpoolingOutputs
                names = ["out"; "indices"; "size"];
            elseif isa(layer, 'nnet.cnn.layer.DepthSliceLayer')
                n = layer.NumOutputs;
                names = "out" + (1:n)';
            elseif isa(internalLayer, 'nnet.internal.cnn.layer.OutputLayer')
                names = string.empty;
            else
                names = "out";
            end
        end
        
        function layer = get.ExternalLayer(this)
            layer = this.TargetLayer;
        end
        
        function internalLayer = get.InternalLayer(this)
            internalLayer = iGetInternalLayers(this.TargetLayer);
            internalLayer = internalLayer{1};
        end
        
    end
    
    methods (Access = protected)
        
        % We need to define getOneLineDisplay because it is abstract in
        % nnet.cnn.layer.Layer
        function [description, type] = getOneLineDisplay(~)
            description = '';
            type = '';
        end
        
    end
    
    methods (Access = private)
        function p = getPropertyNameFromGroup(this, type)
            properties = this.getPropertyGroupFor(type);
            if ~isempty(properties)
                p = string(properties.PropertyList(:));
            else
                p = string.empty();
            end
        end
        function g = getPropertyGroupFor(this, type)
            switch type
                case "HyperParameters"
                    title = iGetStringMessage('nnet_cnn:layer:mixin:ScalarLayerDisplay:HyperparametersGroupTitle');
                case "LearnableParameters"
                    title = iGetStringMessage('nnet_cnn:layer:mixin:ScalarLayerDisplay:LearnableParametersGroupTitle');
                case "DynamicParameters"
                    title = iGetStringMessage('nnet_cnn:layer:mixin:ScalarLayerDisplay:DynamicParametersGroupTitle');
                case "StandadProperties"
                    title = "";
            end
            
            groups = this.TargetLayer.getPropertyGroups();
            matchesTitle = {groups.Title} == string(title);
            g = groups(matchesTitle);
        end
    end
    
end

function stringMessage = iGetStringMessage(id, varargin)
stringMessage = getString( message( id, varargin{:} ) );
end

function o = iGetInternalLayers(layer)
    o = nnet.internal.cnn.layer.util.ExternalInternalConverter.getInternalLayers(layer);
end
