classdef(Abstract) Layer <  matlab.mixin.CustomDisplay & nnet.cnn.layer.mixin.ScalarLayerDisplay & matlab.mixin.Heterogeneous
    % Layer   Interface for network layers
    %
    % To define the architecture of a network, create a vector of layers, e.g.,
    %
    %   layers = [
    %       imageInputLayer([28 28 3])
    %       convolution2dLayer([5 5], 10)
    %       reluLayer()
    %       fullyConnectedLayer(10)
    %       softmaxLayer()
    %       classificationLayer()
    %   ];
    %
    % See also nnet.cnn.layer, trainNetwork, imageInputLayer,
    % sequenceInputLayer, convolution2dLayer, reluLayer, maxPooling2dLayer,
    % averagePooling2dLayer, batchNormalizationLayer, lstmLayer,
    % bilstmLayer, fullyConnectedLayer, softmaxLayer, classificationLayer,
    % regressionLayer.
    
    %   Copyright 2015-2017 The MathWorks, Inc.
    
    properties (Abstract)
        % Name   A name for the layer
        Name
    end
    
    properties (Hidden, Access = protected)
        Version = 1
    end
    
    methods (Hidden, Static)        
        function internalLayers = getInternalLayers(layers)
            internalLayers = nnet.internal.cnn.layer.util.ExternalInternalConverter.getInternalLayers( layers );
        end
    end
    
    methods (Hidden)
        function displayAllProperties(this)
            proplist = properties( this );
            matlab.mixin.CustomDisplay.displayPropertyGroups( ...
                this, ...
                this.propertyGroupGeneral( proplist ) );
        end
    end
    
    methods (Abstract, Access = protected)
        [description, type] = getOneLineDisplay(layer)
    end
    
    methods (Sealed, Access = protected)
        function displayNonScalarObject(layers)
            % displayNonScalarObject   Display function for non scalar
            % objects
            if isvector( layers )
                header = sprintf( '  %s\n', getVectorHeader( layers ) );
                disp( header )
                layers.displayOneLines()
            else
                fprintf( '  %s', getArrayHeader( layers ) )
                layers.displayOnlyTypes()
            end
        end
        
        function displayEmptyObject(layers)
            displayEmptyObject@matlab.mixin.CustomDisplay(layers);
        end
    end
    
    methods (Static, Access = protected)
        function defaultObject = getDefaultScalarElement() %#ok<STOUT>
            exception = MException( message( 'nnet_cnn:layer:Layer:NoDefaultScalarElement' ) );
            throwAsCaller( exception );
        end
    end
    
    methods (Sealed, Access = private)
        function header = getVectorHeader( layers )
            % getVectorHeader   Return the header to be displayed for a
            % vector of layers
            sizeString = sprintf( '%dx%d', size( layers ) );
            className = matlab.mixin.CustomDisplay.getClassNameForHeader( layers );
            header = iGetStringMessage( ...
                'nnet_cnn:layer:Layer:VectorHeader', ...
                sizeString, ...
                className );
        end
        
        function header = getArrayHeader( layers )
            % getArrayHeader   Return the header to be displayed for an
            % array of layers of size >= 2
            
            sizeString = iSizeToString( size(layers) );
            className = matlab.mixin.CustomDisplay.getClassNameForHeader( layers );
            header = iGetStringMessage( ...
                'nnet_cnn:layer:Layer:ArrayHeader', ...
                sizeString, ...
                className );
        end
        
        function displayOneLines(layers)
            % displayOneLines   Display one line for each layer in the
            % vector
            names = iGetLayersNames( layers );
            maxNameLength = iMaxLength( names );
            [descriptions, types] = iGetOneLineDisplay( layers );
            maxTypeLength= iMaxLength( types );
            for idx=1:numel(layers)
                iDisplayOneLine( idx, ...
                    names{idx}, maxNameLength, ...
                    types{idx}, maxTypeLength, ...
                    descriptions{idx} )
            end
        end
        
        function displayOnlyTypes(layers)
            % displayOnlyTypes   Display types for each layer in the vector
            % in brackets. If there are more than 3 layers, the rest will
            % be substituted by '...'
            [~, types] = iGetOneLineDisplay( layers );
            fprintf(' (')
            fprintf('%s', types{1})
            numLayers = numel( layers );
            for idx=2:min( 3, numLayers )
                fprintf(', %s', types{idx})
            end
            if numLayers>3
                fprintf(', ...')
            end
            fprintf(')\n')
        end
    end
end

function sizeString = iSizeToString( sizeVector )
% iSizeToString   Convert a size vector into a formatted size string where
% each dimension is separated by 'x'.
sizeString = sprintf( '%d', sizeVector(1) );
for i=2:numel(sizeVector)
    sizeString = sprintf( '%sx%d', sizeString, sizeVector(i) );
end

end

function stringMessage = iGetStringMessage(id, varargin)
stringMessage = getString( message( id, varargin{:} ) );
end

function res = iMaxLength(strings)
% iMaxLength   Return the maximum string length of a cell array of strings
res = max(cellfun('length', strings));
end

function names = iGetLayersNames(layers)
% iGetLayersNames   Get the name for each layer.
names = arrayfun(@(x)iWrapApostrophe(x.Name), layers, 'UniformOutput', false);
end

function string = iWrapApostrophe(string)
string = ['''' string ''''];
end

function [descriptions, types] = iGetOneLineDisplay(layers)
[descriptions, types] = arrayfun( @(x)x.getOneLineDisplay, layers, 'UniformOutput', false );
end

function iDisplayOneLine(idx, name, maxNameWidth, type, maxTypeWidth, oneLineDescription)
% iDisplayOneLine   Display one line for a layer formatted in a table-like
% style.

fprintf( '    %2i   %-*s   %-*s   %s\n', ...
    idx, ...
    maxNameWidth, name, ...
    maxTypeWidth, type, ...
    oneLineDescription )

end
