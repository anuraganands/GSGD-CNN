classdef TestCase < matlab.unittest.TestCase
    % TestCase   TestCase class to check nnet.layer.Layer behaviour
    %
    %   To check the validity of a layer, use checkLayer
    
    %   Copyright 2017-2018 The MathWorks, Inc.
    
    properties (SetAccess = private)
        % Layer (nnet.layer.Layer)   Layer under test
        Layer
        
        % LayerInformation (nnet.checklayer.MetaLayer) Information about
        % layer
        LayerInformation
        
        % InputData (containers.Map)   Map that stores the input data to
        % test the layer for 'one' and 'multiple' observations          
        InputData
        
        % ObservationDimension (integer)   Dimension in the data that 
        % represents observations (is empty if not specified by the user)
        ObservationDimension = []
    end
    
    properties(TestParameter)
        Type = {'single', 'double','gpuArray'};
        Observations = {'one', 'multiple'}
    end
    
    methods
        function test = TestCase(layer, inputSize, observationDimension)
            test.Layer = layer;
            test.LayerInformation = nnet.checklayer.MetaLayer( layer );
            test.InputData = iGenerateInputData( inputSize, observationDimension );
            test.ObservationDimension = observationDimension;
        end
    end
    
    methods(Test)
        function predictSignatureIsWellDefined(test)
            % predictSignatureIsWellDefined   Test that the signature for
            % the 'predict' method of the layer has exactly 2 input
            % arguments and 1 output argument
            expectedNumInputArgs = 2;
            expectedNumOutputArgs = 1;
            methodUnderTest = test.LayerInformation.Predict;
            
            actualInputArgs = iGetInputs( methodUnderTest );
            actualOutputArgs = iGetOutputs( methodUnderTest );
            
            test.verifyNumElements(actualInputArgs, expectedNumInputArgs, ...
                getString(message('nnet_cnn:nnet:checklayer:TestCase:InvalidInputArguments','predict')));
            test.verifyNumElements(actualOutputArgs, expectedNumOutputArgs, ...
                getString(message('nnet_cnn:nnet:checklayer:TestCase:InvalidOutputArguments','predict')));
        end
        
        function forwardSignatureIsWellDefined(test)
            % forwardSignatureIsWellDefined   Test that the signature for
            % the 'forward' method of the layer has exactly 2 input
            % arguments and 2 output arguments
            expectedNumInputArgs = 2;
            expectedNumOutputArgs = 2;
            methodUnderTest = test.LayerInformation.Forward;
            
            actualInputArgs = iGetInputs( methodUnderTest );
            actualOutputArgs = iGetOutputs( methodUnderTest );
            
            test.verifyNumElements(actualInputArgs, expectedNumInputArgs, ...
                getString(message('nnet_cnn:nnet:checklayer:TestCase:InvalidInputArguments','forward')));
            test.verifyNumElements(actualOutputArgs, expectedNumOutputArgs, ...
                getString(message('nnet_cnn:nnet:checklayer:TestCase:InvalidOutputArguments','forward')));
        end
        
        function backwardSignatureIsWellDefined(test)
            % backwardSignatureIsWellDefined   Test that the signature for
            % the 'backward' method of the layer has exactly 5 input
            % arguments and 1 + number of learnable parameters output
            % arguments
            expectedNumInputArgs = 5;
            numLearnableParameters = numel( test.LayerInformation.ParametersNames );
            expectedNumOutputArgs = 1 + numLearnableParameters;
            methodUnderTest = test.LayerInformation.Backward;
            
            actualInputArgs = iGetInputs( methodUnderTest );
            actualOutputArgs = iGetOutputs( methodUnderTest );
            
            test.verifyNumElements(actualInputArgs, expectedNumInputArgs, ...
                getString(message('nnet_cnn:nnet:checklayer:TestCase:InvalidInputArguments','backward')));
            test.verifyNumElements(actualOutputArgs, expectedNumOutputArgs, ...
                getString(message('nnet_cnn:nnet:checklayer:TestCase:InvalidOutputArguments','backward')));
        end
        
        function predictDoesNotError(test,Observations)
            % predictDoesNotError   Test that 'predict' can be used with no
            % errors
            
            test.assumeObservationDimIsSpecified(Observations);
            
            inputData = test.InputData(Observations);
            test.setLayerPrecision('single');
            
            fcn = @()test.Layer.predict(inputData);
            numArgout = 1;
            
            test.verifyThat(fcn, iDoesNotError('predict',numArgout))
        end
        
        function forwardDoesNotError(test,Observations)
            % forwardDoesNotError   Test that 'forward' can be used with no
            % errors
            
            % If 'forward' was not implemented in the layer then do not run
            % this test
            if ~test.LayerInformation.IsForwardDefined
                % Nothing to test, 'forward' was defined in the superclass
                % and not overridden
                return
            end
            
            test.assumeObservationDimIsSpecified(Observations);
            
            inputData = test.InputData(Observations);
            test.setLayerPrecision('single');
            
            fcn = @()test.Layer.forward(inputData);
            numArgout = 2;
            
            test.verifyThat(fcn, iDoesNotError('forward',numArgout))
        end
        
        function backwardDoesNotError(test,Observations)
            % backwardDoesNotError   Test that 'backward' can be used with
            % no errors
            
            test.assumeObservationDimIsSpecified(Observations);
            
            inputData = test.InputData(Observations);
            test.setLayerPrecision('single');
            
            [Z, memory] = test.tryForward( inputData );
            
            dLdZ = iGenerateSingleData( size(Z) );
            
            fcn = @()test.Layer.backward( inputData, Z, dLdZ, memory );
            numArgout = numel(test.LayerInformation.ParametersValues) + 1;
            
            test.verifyThat(fcn, iDoesNotError('backward',numArgout))
        end
        
        function backwardIsConsistentInSize(test,Observations)
            % backwardIsConsistentInSize   Test that the output of
            % 'backward' is consistent in size. Namely, dLdX should be the
            % same size as X and each dLdW should be the same size as W
            
            test.assumeObservationDimIsSpecified(Observations);
            
            inputData = test.InputData(Observations);
            test.setLayerPrecision('single');
            
            [Z, memory] = test.tryForward( inputData );
            
            dLdZ = iGenerateSingleData( size(Z) );
            
            [dLdX, dLdW{1:numel(test.LayerInformation.ParametersValues)}] = ...
                test.tryBackward( inputData, Z, dLdZ, memory );
            
            test.assertdLdXSameSizeAsX( dLdX, inputData );
            test.assertdLdWSameSizeAsWForEachLearnableParam( dLdW );
        end
        
        function predictIsConsistentInType(test,Type)
            % predictIsConsistentInType   Test that the output of
            % 'predict' is consistent in type. Namely, Z should be the
            % same type as X.
            
            test.assumeGPUisAvailableForGpuArray(Type);
            
            castFcn = iCastFcnFromType(Type);
            inputData = castFcn(test.InputData('one'));
            test.setLayerPrecision(castFcn);
            
            Z = test.tryPredict(inputData);
            
            test.verifyThat( Z, iIsOfSameTypeAs(inputData), ...
                getString(message('nnet_cnn:nnet:checklayer:TestCase:IncorrectTypeZ', 'predict')) );
        end
        
        function forwardIsConsistentInType(test,Type)
            % forwardIsConsistentInType   Test that the output of
            % 'forward' is consistent in type. Namely, Z should be the
            % same type as X.
            
            % If 'forward' was not implemented in the layer then do not run
            % this test
            if ~test.LayerInformation.IsForwardDefined
                % Nothing to test, forward was defined in the superclass
                % and not overridden
                return
            end
            
            test.assumeGPUisAvailableForGpuArray(Type);
            
            castFcn = iCastFcnFromType(Type);
            inputData = castFcn(test.InputData('one'));
            test.setLayerPrecision(castFcn);
            
            Z = test.tryForward( inputData );
            
            test.verifyThat( Z, iIsOfSameTypeAs(inputData), ...
                getString(message('nnet_cnn:nnet:checklayer:TestCase:IncorrectTypeZ', 'forward')) );
        end
        
        function backwardIsConsistentInType(test,Type)
            % backwardIsConsistentInType   Test that the output of
            % 'backward' is consistent in type. Namely, dLdX should be the
            % same type as X and each dLdW should be the same type as W.
            
            test.assumeGPUisAvailableForGpuArray(Type);
            
            castFcn = iCastFcnFromType(Type);
            inputData = castFcn(test.InputData('one'));
            test.setLayerPrecision(castFcn);
            
            [Z, memory] = test.tryForward( inputData );
            
            dLdZ = castFcn(iGenerateSingleData(size(Z)));
            [dLdX, dLdW{1:numel(test.LayerInformation.ParametersValues)}] = ...
                test.tryBackward( inputData, Z, dLdZ, memory );
            
            test.verifyThat( dLdX, iIsOfSameTypeAs(inputData), ...
                getString(message('nnet_cnn:nnet:checklayer:TestCase:IncorrectTypedLdX')) );
            
            test.verifySameTypeForEachdLdW( dLdW, inputData );
        end
        
        function gradientsAreNumericallyCorrect(test)
            % gradientsAreNumericallyCorrect   Test that the gradients
            % computed in 'backward' are numerically correct
            
            relTol = 1e-6;
            absTol = 1e-6;
            mixedTol = iRelativeTolerance(relTol) | iAbsoluteTolerance(absTol);
            numParameters = numel( test.LayerInformation.ParametersNames );
            
            test.setLayerPrecision('double');
            X0 = double(test.InputData('one'));
            [Z, memory] = test.tryForward(X0);
            % Pertub dLdZ so that it is not equal to X0
            dLdZ = iGenerateDoubleData( size(Z) ) * 0.9;
            
            [actual_dLdX, actual_dLdW{1:numParameters}] = ...
                test.tryBackward(X0, Z, dLdZ, memory);
            
            forwardWithNewX = @(X)forward(test.Layer, X);
            expected_dLdX = iNumericGradient(forwardWithNewX, X0, dLdZ);
            
            test.verifyThat( actual_dLdX, ...
                iIsEqualTo(expected_dLdX, 'Within', mixedTol), ...
                getString(message('nnet_cnn:nnet:checklayer:TestCase:IncorrectGradientLdX')) );
            
            for ii=1:numParameters
                paramName = test.LayerInformation.ParametersNames{ii};
                W0 = test.Layer.(paramName);
                forwardWithNewWeights = @(W)assignWeightsAndForward(test.Layer,paramName,W,X0);
                expected_dLdW = iNumericGradient(forwardWithNewWeights, W0, dLdZ);
                
                test.verifyThat( actual_dLdW{ii}, iIsEqualTo(expected_dLdW, 'Within', mixedTol), ...
                    getString(message('nnet_cnn:nnet:checklayer:TestCase:IncorrectGradientdLdW',paramName)) );
            end
            
            function Z = assignWeightsAndForward(layer,param,W,X)
                layer.(param) = W;
                Z = forward(layer,X);
            end
        end
    end
    
    methods(Access = private)
        function setLayerPrecision(test,type)
            % setLayerPrecision   Cast all learnable parameters of the
            % layer to the given type.
            if isa(type,'function_handle')
                castFcn = type;
            else
                castFcn = @(x) cast(x,type);
            end
            numParameters = numel( test.LayerInformation.ParametersNames );
            for ii=1:numParameters
                paramName = test.LayerInformation.ParametersNames{ii};
                test.Layer.(paramName) = castFcn(test.Layer.(paramName));
            end
        end
        
        function [Z, memory] = tryForward( test, inputData )
            try
                [Z, memory] = test.Layer.forward( inputData );
            catch
                % 'forward' errored out, fail the test
                if test.LayerInformation.IsForwardDefined
                    methodThatErrored = 'forward';
                else
                    methodThatErrored = 'predict';
                end
                test.assumeFail( getString(message('nnet_cnn:nnet:checklayer:TestCase:SkipTestMethodErrored', methodThatErrored)) )
            end
        end
        
        function Z = tryPredict( test, inputData )
            try
                Z = test.Layer.predict( inputData );
            catch
                % 'predict' errored out, fail the test
                test.assumeFail( getString(message('nnet_cnn:nnet:checklayer:TestCase:SkipTestMethodErrored', 'predict')) )
            end
        end
        
        function varargout = tryBackward( test, varargin )
            try
                [varargout{1:nargout}] = test.Layer.backward( varargin{:} );
            catch
                % 'backward' errored out, fail the test
                test.assumeFail( getString(message('nnet_cnn:nnet:checklayer:TestCase:SkipTestMethodErrored', 'backward')) )
            end
        end
        
        function assertdLdXSameSizeAsX( test, dLdX, X )
            dLdXSize = size(dLdX);
            XSize = size(X);
            test.verifyEqual( dLdXSize, XSize, getString(message('nnet_cnn:nnet:checklayer:TestCase:IncorrectSizedLdX')) );
        end
        
        function assertdLdWSameSizeAsWForEachLearnableParam( test, dLdW )
            for ii=1:numel(dLdW)
                test.assertdLdWSameSizeAsW( dLdW{ii}, test.LayerInformation.ParametersValues{ii}, test.LayerInformation.ParametersNames{ii} )
            end
        end
        
        function assertdLdWSameSizeAsW( test, dLdW, W, parameterName )
            WSize = size( W );
            dLdWSize = size( dLdW );
            test.verifyEqual( dLdWSize, WSize, getString(message('nnet_cnn:nnet:checklayer:TestCase:IncorrectSizedLdW',parameterName)) );
        end
        
        function verifySameTypeForEachdLdW( test, dLdW, X )
            for ii=1:numel(dLdW)
                parameterName = test.LayerInformation.ParametersNames{ii};
                test.verifyThat( dLdW{ii}, iIsOfSameTypeAs(X), ...
                    getString(message('nnet_cnn:nnet:checklayer:TestCase:IncorrectTypedLdW',parameterName)));
            end
        end
        
        function assumeGPUisAvailableForGpuArray( test, type )
            if strcmp(type,'gpuArray')
                hasGPU = nnet.internal.cnn.util.isGPUCompatible();
                test.assumeTrue( hasGPU, getString(message('nnet_cnn:nnet:checklayer:TestCase:NoGPUAvailable')) );
            end
        end
        
        function assumeObservationDimIsSpecified( test, observation )
            if strcmp(observation,'multiple')
                isObsDimSpecified = ~isempty(test.ObservationDimension);
                test.assumeTrue( isObsDimSpecified, getString(message('nnet_cnn:nnet:checklayer:TestCase:ObsDimNotSpecified')) );
            end
        end
        
    end
end

function n = iGetInputs( currentMethod )
n = currentMethod.InputNames;
end

function n = iGetOutputs( currentMethod )
n = currentMethod.OutputNames;
end

function data = iGenerateSingleData( sz )
% iGenerateSingleData   Generate single data using halton sequences
data = nnet.checklayer.halton( sz, 1000, 101 );
data = cast( data, 'single' );
end

function data = iGenerateDoubleData( sz )
% iGenerateDoubleData   Generate double data using halton sequences
data = nnet.checklayer.halton( sz, 1000, 101 );
data = cast( data, 'double' );
end

function inputData = iGenerateInputData( inputSize, observationDimension )
% Generate data for a single observation
oneObsSize = iSetDimension(inputSize,observationDimension,1);
oneObsData = iGenerateSingleData(oneObsSize);
% Generat data for two observations
multipleObsSize = iSetDimension(inputSize,observationDimension,2);
multipleObsData = iGenerateSingleData(multipleObsSize);
% Store data in a Map container
inputData = containers.Map( {'one', 'multiple'}, {oneObsData, multipleObsData} );
end

function outputSize = iSetDimension(inputSize,dimension,value)
outputSize = inputSize;
numOutputDims = max( numel(inputSize), dimension );
outputSize(end+1:numOutputDims) = 1;
outputSize(dimension) = value;
end

function castFcn = iCastFcnFromType(type)
if strcmp(type,'gpuArray')
    castFcn = @(x) gpuArray(single(x));
else
    castFcn = @(x) cast(x,type);
end
end

function constraint = iDoesNotError(varargin)
constraint = nnet.checklayer.constraints.DoesNotThrowErrors(varargin{:});
end

function constraint = iIsOfSameTypeAs(value)
constraint = nnet.checklayer.constraints.IsOfSameTypeAs(value);
end

function constraint = iIsEqualTo(varargin)
constraint = matlab.unittest.constraints.IsEqualTo(varargin{:});
end

function constraint = iRelativeTolerance(varargin)
constraint = matlab.unittest.constraints.RelativeTolerance(varargin{:});
end

function constraint = iAbsoluteTolerance(varargin)
constraint = matlab.unittest.constraints.AbsoluteTolerance(varargin{:});
end

function dLdIn = iNumericGradient(varargin)
dLdIn = nnet.checklayer.numericDiff5Point(varargin{:});
end