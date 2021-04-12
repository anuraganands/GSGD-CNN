classdef(Abstract) ScalarLayerDisplay
    % ScalarLayerDisplay   Scalar layer display customization class
    %   This class defines a set of utilities for customizing the
    %   appearance of a scalar layer display.
    
    %   Copyright 2016-2017 The MathWorks, Inc.
    
    methods (Sealed, Access = protected)
        function pgroup = propertyGroupGeneral( ~, properties )
            pgroup = iPropertyGroup( '', properties );
        end
        
        function pgroup = propertyGroupHyperparameters( ~, properties )
            title = iGetStringMessage( 'nnet_cnn:layer:mixin:ScalarLayerDisplay:HyperparametersGroupTitle' );
            pgroup = iPropertyGroup( title, properties );
        end
        
        function pgroup = propertyGroupLearnableParameters( ~, properties )
            title = iGetStringMessage( 'nnet_cnn:layer:mixin:ScalarLayerDisplay:LearnableParametersGroupTitle' );
            pgroup = iPropertyGroup( title, properties );
        end
        
        function pgroup = propertyGroupDynamicParameters( ~, properties )
            title = iGetStringMessage( 'nnet_cnn:layer:mixin:ScalarLayerDisplay:DynamicParametersGroupTitle' );
            pgroup = iPropertyGroup( title, properties );
        end
        
        function footer = createShowAllPropertiesFooter( ~, variableName )
            useHotlinks = feature( 'hotlinks' ) && ~isdeployed();
            if ~isempty(variableName) && useHotlinks
                % The text is the message the user will see on screen.
                text = iGetStringMessage( 'nnet_cnn:layer:mixin:ScalarLayerDisplay:ShowAllProperties' );
                
                % The command is what get executed when the user clicks on the link
                command = sprintf( 'try, displayAllProperties(%s), end', variableName);
                
                % ... then the text needs to be wrapped in an HREF to make the link
                footer = sprintf( '  <a href="matlab:%s">%s</a>\n', command, text );
            else
                footer = iGetStringMessage( 'nnet_cnn:layer:mixin:ScalarLayerDisplay:NoHotlinkFooter' );
                footer = sprintf( '%s\n', footer );
            end
        end
    end
    
    methods (Abstract)
        % displayAllProperties   Display all the properties of a layer
        displayAllProperties(layer)
    end
end

function pgroup = iPropertyGroup( title, properties )
% iPropertyGroup   Create a property group from title and properties.
pgroup = matlab.mixin.util.PropertyGroup(properties,title);
end

function stringMessage = iGetStringMessage(id, varargin)
stringMessage = getString( message( id, varargin{:} ) );
end