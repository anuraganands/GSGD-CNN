classdef CustomLayerVerifier
    % CustomLayerVerifier   Class that holds checks for a custom layer
    
    %   Copyright 2017 The MathWorks, Inc.
    
    properties
        % LayerClass
        LayerClass
        
        % LearnableParametersNames
        LearnableParametersNames
    end
    
    methods
        function verifyPredictType( this, X, Z )
            % verifyPredictType   Verify the type of the output returned by
            % a custom layer at the time predict is called
            iAssertZSameTypeAsXInPredict( Z, X, this.LayerClass );
        end
        
        function verifyForwardType( this, X, Z )
            % verifyForwardType   Verify the type of the output returned by
            % a custom layer at the time forward is called
            iAssertZSameTypeAsXInForward( Z, X, this.LayerClass );
        end
        
        function verifyBackwardSize( this, X, learnableParameters, dLdX, dLdW )
            % verifyBackwardSize   Verify the size of the output returned
            % by a custom layer at the time backward is called
            iAssertdLdXSameSizeAsX( dLdX, X, this.LayerClass );
            verifyWeightGradients = nargin > 4;
            if verifyWeightGradients
                iAssertdLdWSameSizeAsWForEachLearnableParam( dLdW, learnableParameters, this.LearnableParametersNames, this.LayerClass );
            end
        end
        
        function verifyBackwardType( this, X, dLdX, dLdW)
            % verifyBackwardType   Verify the type of the output returned
            % by a custom layer at the time backward is called
            iAssertdLdXSameTypeAsX( dLdX, X, this.LayerClass );
            verifyWeightGradients = nargin > 3;
            if verifyWeightGradients
                iAssertdLdWSameTypeAsXForEachLearnableParam( dLdW, X, this.LearnableParametersNames, this.LayerClass );
            end
        end
        
        function verifyForwardLossSize( ~, loss )
            % verifyForwardLossSize   Verify the size of the loss returned
            % by a custom output layer at the time forwardLoss is called
            iAssertScalarLoss( loss );
        end
        
        function verifyForwardLossType( ~, X, loss )
            % verifyForwardLossType   Verify the type of the loss returned
            % by a custom output layer at the time forwardLoss is called
            iAssertLossSameTypeAsX( loss, X );
            
        end
        
        function verifyBackwardLossSize( ~, X, dLdX )
            % verifyBackwardLoss   Verify the size of the output returned
            % by a custom output layer at the time backwardLoss is called
            iAssertdLdXSameSizeAsXInBackwardLoss( dLdX, X );
        end
        
        function verifyBackwardLossType( ~, X, dLdX )
            % verifyBackwardLossType   Verify the type of the output
            % returned by a custom output layer at the time backwardLoss
            % is called
            iAssertdLdXSameTypeAsXInBackwardLoss( dLdX, X );
        end
        
    end
    
    methods (Static)
        function validateMethodSignatures( aCustomLayer, aLayerIndex )
            % validateMethodSignatures   Validate the signatures of
            % predict, forward and backward methods of a custom layer.
            %
            % Inputs:
            %   aCustomLayer - a custom layer to validate
            %   aLayerIndex  - the index of the layer in the array. This
            %                  will be used to build a useful error message
            if iIsACustomOutputLayer( aCustomLayer )
                [forwardLoss, backwardLoss] = iGetOutputLayerMethods( aCustomLayer );
                
                iAssertCorrectForwardLossSignature( forwardLoss, aLayerIndex );
                iAssertCorrectBackwardLossSignature( backwardLoss, aLayerIndex );
            else
                [predict, forward, backward] = iGetMethods( aCustomLayer );
                
                iAssertCorrectPredictSignature( predict, aLayerIndex );
                iAssertCorrectForwardSignature( forward, aLayerIndex );
                
                numLearnableParameters = iValidateLearnableParameters( aCustomLayer, aLayerIndex );
                iAssertCorrectBackwardSignature( backward, numLearnableParameters, aLayerIndex );
            end
        end
    end
end

function tf = iIsACustomOutputLayer( aCustomLayer )
tf = isa( aCustomLayer, 'nnet.layer.ClassificationLayer' ) ...
    || isa( aCustomLayer, 'nnet.layer.RegressionLayer' ) ;
end

function [predict, forward, backward] = iGetMethods( aCustomLayer )
% iGetMethods   Get predict/forward/backward methods from a custom layer
m = metaclass( aCustomLayer );
methodList = m.MethodList;
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

function [forwardLoss, backwardLoss] = iGetOutputLayerMethods( aCustomLayer )
% iGetOutputLayerMethods   Get forwardLoss/backwardLoss methods from a
% custom output layer
m = metaclass( aCustomLayer );
methodList = m.MethodList;
for ii=1:numel(methodList)
    currentMethod = methodList(ii);
    switch currentMethod.Name
        case 'forwardLoss'
            forwardLoss = currentMethod;
        case 'backwardLoss'
            backwardLoss = currentMethod;
        otherwise
    end
end
end

function iAssertCorrectPredictSignature( predict, aLayerIndex )
expectedNargin = 2;
arginErrorID = 'nnet_cnn:internal:cnn:layer:util:CustomLayerVerifier:WrongPredictNargin';
iAssertInputArguments( predict, aLayerIndex, expectedNargin, arginErrorID);

expectedNargout = 1;
argoutErrorID = 'nnet_cnn:internal:cnn:layer:util:CustomLayerVerifier:WrongPredictNargout';
iAssertOutputArguments( predict, aLayerIndex, expectedNargout, argoutErrorID);
end

function iAssertCorrectForwardSignature( forward, aLayerIndex )
expectedNargin = 2;
arginErrorID = 'nnet_cnn:internal:cnn:layer:util:CustomLayerVerifier:WrongForwardNargin';
iAssertInputArguments( forward, aLayerIndex, expectedNargin, arginErrorID);

expectedNargout = 2;
argoutErrorID = 'nnet_cnn:internal:cnn:layer:util:CustomLayerVerifier:WrongForwardNargout';
iAssertOutputArguments( forward, aLayerIndex, expectedNargout, argoutErrorID);
end

function iAssertCorrectBackwardSignature( backward, numLearnableParams, aLayerIndex )
expectedNargin = 5;
arginErrorID = 'nnet_cnn:internal:cnn:layer:util:CustomLayerVerifier:WrongBackwardNargin';
iAssertInputArguments( backward, aLayerIndex, expectedNargin, arginErrorID);

expectedNargout = 1 + numLearnableParams;
if numLearnableParams > 0
    argoutErrorID = 'nnet_cnn:internal:cnn:layer:util:CustomLayerVerifier:WrongBackwardNargoutWithLearnableParams';
    iAssertOutputArguments( backward, aLayerIndex, expectedNargout, argoutErrorID, numLearnableParams);
else
    argoutErrorID = 'nnet_cnn:internal:cnn:layer:util:CustomLayerVerifier:WrongBackwardNargout';
    iAssertOutputArguments( backward, aLayerIndex, expectedNargout, argoutErrorID);
end
end

function iAssertCorrectForwardLossSignature( forwardLoss, aLayerIndex )
expectedNargin = 3;
arginErrorID = 'nnet_cnn:internal:cnn:layer:util:CustomLayerVerifier:WrongForwardLossNargin';
iAssertInputArguments( forwardLoss, aLayerIndex, expectedNargin, arginErrorID);

expectedNargout = 1;
argoutErrorID = 'nnet_cnn:internal:cnn:layer:util:CustomLayerVerifier:WrongForwardLossNargout';
iAssertOutputArguments( forwardLoss, aLayerIndex, expectedNargout, argoutErrorID);
end

function iAssertCorrectBackwardLossSignature( backwardLoss, aLayerIndex )
expectedNargin = 3;
arginErrorID = 'nnet_cnn:internal:cnn:layer:util:CustomLayerVerifier:WrongBackwardLossNargin';
iAssertInputArguments( backwardLoss, aLayerIndex, expectedNargin, arginErrorID);

expectedNargout = 1;
argoutErrorID = 'nnet_cnn:internal:cnn:layer:util:CustomLayerVerifier:WrongBackwardLossNargout';
iAssertOutputArguments( backwardLoss, aLayerIndex, expectedNargout, argoutErrorID);
end

function iAssertdLdXSameSizeAsX( dLdX, X, layerClassOrIndex )
dLdXSize = size(dLdX);
XSize = size(X);
if ~isequal( dLdXSize, XSize )
    iThrow( 'nnet_cnn:internal:cnn:layer:util:CustomLayerVerifier:WrongSizeOfdLdX', layerClassOrIndex, iSizeToString(XSize), iSizeToString(dLdXSize) )
end
end

function iAssertdLdWSameSizeAsWForEachLearnableParam( dLdW, learnableParameters, learnableParametersNames, layerClass )
for ii=1:numel(learnableParameters)
    iAssertdLdWSameSizeAsW( dLdW{ii}, learnableParameters(ii).Value, learnableParametersNames{ii}, layerClass )
end
end

function iAssertdLdWSameSizeAsW( dLdW, W, parameterName, layerNameOrIndex )
WSize = size( W );
dLdWSize = size( dLdW );
if ~isequal( dLdWSize, WSize )
    iThrow( 'nnet_cnn:internal:cnn:layer:util:CustomLayerVerifier:WrongSizeOfdLdW', parameterName, layerNameOrIndex, iSizeToString(WSize), iSizeToString(dLdWSize) )
end
end

function iAssertScalarLoss( loss )
if ~isscalar( loss )
    iThrow( 'nnet_cnn:internal:cnn:layer:util:CustomLayerVerifier:ScalarLoss', iSizeToString( size( loss ) ) )
end
end

function iAssertdLdXSameSizeAsXInBackwardLoss( dLdX, X )
dLdXSize = size( dLdX );
XSize = size( X );
if ~isequal( dLdXSize, XSize )
    iThrow( 'nnet_cnn:internal:cnn:layer:util:CustomLayerVerifier:WrongSizeOfdLdXInBackwardLoss', iSizeToString(XSize), iSizeToString(dLdXSize) )
end
end

function iAssertZSameTypeAsXInPredict( Z, X, layerClass )
classX = class(X);
classZ = class(Z);
if ~isequal( classX, classZ )
    iThrow( 'nnet_cnn:internal:cnn:layer:util:CustomLayerVerifier:PredictInvalidType', ...
        layerClass, classX, classZ )
end
if strcmp(classX,'gpuArray')
    classUnderX = classUnderlying(X);
    classUnderZ = classUnderlying(Z);
    if ~isequal( classUnderX, classUnderZ )
        iThrow( 'nnet_cnn:internal:cnn:layer:util:CustomLayerVerifier:PredictInvalidUnderlyingType', ...
            layerClass, classUnderX, classUnderZ )
    end
end
end

function iAssertZSameTypeAsXInForward( Z, X, layerClass )
classX = class(X);
classZ = class(Z);
if ~isequal( classX, classZ )
    iThrow( 'nnet_cnn:internal:cnn:layer:util:CustomLayerVerifier:ForwardInvalidType', layerClass, classX, classZ )
end
if strcmp(classX,'gpuArray')
    classUnderX = classUnderlying(X);
    classUnderZ = classUnderlying(Z);
    if ~isequal( classUnderX, classUnderZ )
        iThrow( 'nnet_cnn:internal:cnn:layer:util:CustomLayerVerifier:ForwardInvalidUnderlyingType', ...
            layerClass, classUnderX, classUnderZ )
    end
end
end

function iAssertdLdXSameTypeAsX( dLdX, X, layerClass )
classX = class(X);
classdLdX = class(dLdX);
if ~isequal( classX, classdLdX )
    iThrow( 'nnet_cnn:internal:cnn:layer:util:CustomLayerVerifier:dLdXInvalidType', layerClass, classX, classdLdX )
end
if strcmp(classX,'gpuArray')
    classUnderX = classUnderlying(X);
    classUnderdLdX = classUnderlying(dLdX);
    if ~isequal( classUnderX, classUnderdLdX )
        iThrow( 'nnet_cnn:internal:cnn:layer:util:CustomLayerVerifier:dLdXInvalidUnderlyingType', ...
            layerClass, classUnderX, classUnderdLdX )
    end
end
end

function iAssertdLdWSameTypeAsXForEachLearnableParam( dLdW, X, learnableParametersNames, layerClass )
classX = class(X);
for ii=1:numel(learnableParametersNames)
    classdLdW = class( dLdW{ii} );
    if ~isequal( classdLdW, classX )
        iThrow( 'nnet_cnn:internal:cnn:layer:util:CustomLayerVerifier:dLdWInvalidType', ...
            learnableParametersNames{ii}, layerClass, classX, classdLdW )
    end
    if strcmp(classX,'gpuArray')
        classUnderX = classUnderlying(X);
        classUnderdLdW = classUnderlying(dLdW{ii});
        if ~isequal( classUnderX, classUnderdLdW )
            iThrow( 'nnet_cnn:internal:cnn:layer:util:CustomLayerVerifier:dLdWInvalidUnderlyingType', ...
                learnableParametersNames{ii}, layerClass, classUnderX, classUnderdLdW )
        end
    end
end
end

function iAssertLossSameTypeAsX( loss, X )
classX = class(X);
classLoss = class(loss);
if ~isequal( classLoss, classX )
    iThrow( 'nnet_cnn:internal:cnn:layer:util:CustomLayerVerifier:LossInvalidType', classX, classLoss )
end
if strcmp(classX,'gpuArray')
    classUnderX = classUnderlying(X);
    classUnderLoss = classUnderlying(loss);
    if ~isequal( classUnderX, classUnderLoss )
        iThrow( 'nnet_cnn:internal:cnn:layer:util:CustomLayerVerifier:LossInvalidUnderlyingType', ...
            classUnderX, classUnderLoss )
    end
end
end

function iAssertdLdXSameTypeAsXInBackwardLoss( dLdX, X )
classX = class(X);
classdLdX = class( dLdX );
if ~isequal( classdLdX, classX )
    iThrow( 'nnet_cnn:internal:cnn:layer:util:CustomLayerVerifier:dLdXBackwardLossInvalidType', classX, classdLdX )
end
if strcmp(classX,'gpuArray')
    classUnderX = classUnderlying(X);
    classUnderdLdX = classUnderlying(dLdX);
    if ~isequal( classUnderX, classUnderdLdX )
        iThrow( 'nnet_cnn:internal:cnn:layer:util:CustomLayerVerifier:dLdXBackwardLossInvalidUnderlyingType', ...
            classUnderX, classUnderdLdX )
    end
end
end

function n = iNargin( currentMethod )
n = numel(currentMethod.InputNames);
end

function n = iNargout( currentMethod )
n = numel(currentMethod.OutputNames);
end

function numLearnableParameters = iValidateLearnableParameters( aCustomLayer, aLayerIndex )
% iValidateLearnableParameters   Make sure learnable parameters are valid
% and return the total number of learnable parameters in the layer
m = metaclass( aCustomLayer );
propertyList = m.PropertyList;
numLearnableParameters = 0;
for ii=1:numel(propertyList)
    if iIsLearnableProperty( propertyList(ii) )
        iAssertValidLearnableParameter( propertyList(ii), aLayerIndex )
        numLearnableParameters = numLearnableParameters + 1;
    end
end
end

function tf = iIsLearnableProperty( prop )
% iIsLearnableProperty   Return true if prop is a property with "Learnable"
% tag. Since we might assess a layer that does not implement the learnable
% metaclass, we need to make sure that the "Learnable" tag exists before
% checking if it is set to true or false
tf = isprop(prop, 'Learnable') && prop.Learnable;
end

function iAssertValidLearnableParameter( prop, aLayerIndex )
% iAssertValidLearnableParameter   Assert that prop is a valid learnable
% parameter property. To be valid, it needs to have set and get access
% public, and not being constant
if ~iIsPublic( prop.GetAccess ) || ~iIsPublic( prop.SetAccess )
    iThrow( 'nnet_cnn:internal:cnn:layer:util:CustomLayerVerifier:NonPublicLearnableParam', prop.Name, aLayerIndex )
end
if prop.Constant
    iThrow( 'nnet_cnn:internal:cnn:layer:util:CustomLayerVerifier:ConstantLearnableParam', prop.Name, aLayerIndex )
end
end

function tf = iIsPublic( access )
% An access in a metaclass can be public if it's either 'public' or 'none'
tf = isequal( access, 'public' ) || isequal( access, 'none' );
end

function iAssertInputArguments( method, aLayerIndex, expectedNargin, errorID)
% iAssertInputArguments   Assert that method has expectedNargin number of
% input arguments. If not, throw errorID error
actualNargin = iNargin( method );
if actualNargin ~= expectedNargin
    iThrow( errorID, aLayerIndex, expectedNargin, actualNargin );
end
end

function iAssertOutputArguments( method, aLayerIndex, expectedNargout, errorID, varargin )
% iAssertOutputArguments   Assert that method has expectedNargout number of
% output arguments. If not, throw errorID error using varargin additional
% parameters
actualNargout = iNargout( method );
if actualNargout ~= expectedNargout
    iThrow( errorID, aLayerIndex, expectedNargout, actualNargout, varargin{:} );
end
end

function sizeString = iSizeToString( sizeVector )
% i3DSizeToString   Convert a size retrieved by "size" into a string
% separated by "x".
sizeString = join( string( sizeVector ), "x" );
end

function iThrow( msg, varargin )
error( message( msg, varargin{:} ) )
end