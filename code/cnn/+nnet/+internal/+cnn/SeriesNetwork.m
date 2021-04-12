classdef SeriesNetwork < nnet.internal.cnn.TrainableNetwork
    % SeriesNetwork   Class for a series convolutional neural network
    %
    %   A series network is always composed by an input layer, some middle
    %   layers, an output layer and a loss layer.
    %   Consistency of the layers and their conformity to this scheme has 
    %   to be checked outside the network.
    
    %   Copyright 2015-2018 The MathWorks, Inc.
    
    properties
        % Layers    Layers of the networks
        %           (cell array of nnet.internal.cnn.layer.Layer)
        Layers = cell.empty;
    end
    
    properties (Dependent, SetAccess = private)
        % LearnableParameters    Learnable parameters of the networks
        %                        (vector of nnet.internal.cnn.layer.LearnableParameter)
        LearnableParameters
    end
    
    methods
        function this = SeriesNetwork(layers)
            % SeriesNetwork     Constructor for SeriesNetwork class
            %
            %   Create a series network with a cell array of
            %   nnet.internal.cnn.layer.Layer
            if nargin
                this.Layers = layers;
            end
        end
        
        function output = predict(this, data)
            % predict   Predict a response based on the input data
            indexOutputLayer = this.indexOutputLayer();
            output = activations(this, data, indexOutputLayer);
        end
        
        function output = activations(this, data, outputLayer)
            
            % Apply transforms to input. These transforms are applied on
            % the CPU prior to moving data to GPU.
            if iInputLayerHasTransforms(this.Layers{1})
                data = apply(this.Layers{1}.Transforms, data);
            end
            
            % Make sure not to call predict on the last layer
            if outputLayer == this.indexOutputLayer
                outputLayer = outputLayer - 1;
            end
            
            output = data;
            for currentLayer = 1:outputLayer
                output = this.Layers{currentLayer}.predict( output );
            end
        end
        
        function [output, states, finalStates] = statefulPredict(this, data, statefulLayers, propagateState)
            % statefulPredict    Forward input data and returns a cell
            % array containing the output of each layer.
            %
            % Inputs
            %   data             - the input data
            %   statefulLayers   - a logical array with "true" values for
            %                      layers with state parameters
            %   propagateState   - a logical which is true when the state
            %                      from one batch should be passed to the
            %                      next
            % Outputs
            %   output       - the output of the network
            %   states       - a cell array containing the states of any
            %                  layer with state parameters, used by the
            %                  updateNetworkState method
            %   finalStates  - a cell array containing the states of any
            %                  layer with state parameters, used by the
            %                  external network predict method
            indexOutputLayer = this.indexOutputLayer();
            states = cell( numel(statefulLayers), 1 );
            finalStates = states;
                        
            output = data;
            for currentLayer = 2:indexOutputLayer-1
                if statefulLayers(currentLayer)
                    % Forward propagate with memory
                    [nextOutput, memory] = this.Layers{currentLayer}.forward( output );
                    
                    % Compute the state used for the updateNetworkState
                    % method
                    states{currentLayer} = computeState( this.Layers{currentLayer}, ...
                        output, nextOutput, memory, propagateState);
                    
                    % Compute final state to be returned by the predict
                    % method
                    if propagateState
                        finalStates{currentLayer} = states{currentLayer};
                    else
                        finalStates{currentLayer} = computeState( this.Layers{currentLayer}, ...
                            output, nextOutput, memory, true);
                    end
                    
                    % Move the output on for the next layer
                    output = nextOutput;
                else
                    % Predict without memory
                    output = this.Layers{currentLayer}.predict( output );
                end
            end
        end
        
        function [layerOutputs, memory] = forwardPropagation(this, data)
            % forwardPropagation    Forward input data and returns a cell
            % array containing the output of each layer.
            %
            % Inputs
            %   data - a gpuArray containing the data
            % Outputs
            %   layerOutputs - a cell array containing the output of the
            %                  forward function on each layer
            %   memory       - a cell array containing the memory output of
            %                  the forward function on each layer
            indexOutputLayer = this.indexOutputLayer();
            layerOutputs = cell(indexOutputLayer-1,1);
            memory = cell(indexOutputLayer-1,1);

            % We can recover GPU memory by gathering the activations and
            % memory cell arrays back to the host.
            function gatherLayerOutputsAndMemory()
                layerOutputs = iGatherGPUCell(layerOutputs);
                memory = iGatherGPUCell(memory);
            end
            
            [layerOutputs{1}, memory{1}] = iExecuteWithStagedGPUOOMRecovery( ...
                @() this.Layers{1}.forward( data ), 2, {@gatherLayerOutputsAndMemory} );
            for currentLayer = 2:indexOutputLayer-1
                [layerOutputs{currentLayer}, memory{currentLayer}] = ...
                    iExecuteWithStagedGPUOOMRecovery( ...
                    @() this.Layers{currentLayer}.forward( layerOutputs{currentLayer-1} ), ...
                    2, {@gatherLayerOutputsAndMemory} );
            end
        end
        
        function [dxLayers, dwLayers] = backwardPropagation(this, layerOutputs, response, memory)
            % backPropagation   Propagate the response from the last layer
            % to the first returning diffs between outputs and inputs
            %
            % Inputs
            %   layerOutputs - a cell array containing the output of the
            %                  forward function on each layer
            %   response     - expected responses
            %   memory       - a cell array containing the memory output of
            %                  the forward function on each layer            
            % Outputs
            %   dxLayers     - cell array containing the derivatives of
            %                  the loss function with respect to the input
            %                  for each layer
            %   dwLayers     - cell array containing the derivatives of
            %                  the loss function with respect to the
            %                  weights for each layer
            
            indexOutputLayer = this.indexOutputLayer();
            
            dxLayers = cell(indexOutputLayer, 1);
            dwLayers = {}; % this will be appended to as we go

            % We can recover GPU memory by gathering the output derivatives
            % back to the host.
            function gatherDiffs()
                dxLayers = iGatherGPUCell(dxLayers);
                dwLayers = iGatherGPUCell(dwLayers);
            end
            
            % Call backward loss on the output layer
            dxLayers{indexOutputLayer} = ...
                iExecuteWithStagedGPUOOMRecovery( ...
                @() this.Layers{indexOutputLayer}.backwardLoss(layerOutputs{end}, response), ...
                1, {@gatherDiffs} );
            
            % Call backward on every other layer, except the first since
            % its delta will be empty
            for el = indexOutputLayer-1:-1:2
                [dxLayers{el},thisDw] = iExecuteWithStagedGPUOOMRecovery( ...
                    @() this.Layers{el}.backward( ...
                    layerOutputs{el-1}, layerOutputs{el}, dxLayers{el+1}, memory{el}), ...
                    2, {@gatherDiffs} );
                % Note that we are building up the gradients backwards
                dwLayers = [thisDw dwLayers]; %#ok<AGROW>
            end
            
        end
        %%
        function [gradients, predictions, states] = computeGradientsForTraining(this, X, Y, needsStatefulTraining, propagateState)
            % computeGradientsForTraining    Computes the gradients of the
            % loss with respect to the learnable parameters, from the
            % network input and response. This is used during training to
            % avoid the need to store intermediate activations and
            % derivatives any longer than is necessary.
            %
            % Inputs
            %   X                      - an array containing the data
            %   Y                      - expected responses
            %   needsStatefulTraining  - logical scalar for each layer
            %                            marking whether the layer needs
            %                            stateful training or not
            %   propagateState         - logical scalar marking whether
            %                            state needs to be propagated or
            %                            not
            %
            % Output
            %   gradients   - cell array of gradients with one element for
            %                 each learnable parameter array
            %   predictions - the output from the last layer, needs to be
            %                 preserved during training to report progress
            %   states      - cell array of state information needed to
            %                 update layer states after gradient computation
            
            indexOutputLayer = this.indexOutputLayer();
            layerOutputs = cell(indexOutputLayer-1,1);
            memory = cell(indexOutputLayer-1,1);
            states = cell(indexOutputLayer-1,1);

            % Set the sequence of memory retrieval functions that can be
            % used by iExecuteWithStagedGPUOOMRecovery to attempt to
            % recover from GPU memory allocation failures.
            %
            % We can recover GPU memory by gathering the current
            % intermediate activations and memory cell arrays back to the
            % host.
            function gatherLayerOutputsAndMemory()
                layerOutputs = iGatherGPUCell(layerOutputs);
                memory = iGatherGPUCell(memory);
            end
            recoveryStrategies = {@gatherLayerOutputsAndMemory};
            %
            % We could also return gradients on the host instead of the GPU
            gradients = {};
            function gatherGradients()
                gradients = iGatherGPUCell(gradients);
            end
            recoveryStrategies = [ recoveryStrategies {@gatherGradients} ];

            % Forward propagation. We need to preserve all intermediate
            % activations at this point.
            layerIsLearning = false(indexOutputLayer, 1);
            numLearnables = 0;
            [layerOutputs{1}, memory{1}] = iExecuteWithStagedGPUOOMRecovery( ...
                @() this.Layers{1}.forward( X ), 2, recoveryStrategies, 1 );
            for currentLayer = 2:indexOutputLayer-1
                
                % Mark whether this layer can learn, for backpropagation
                % optimisation
                learnablesThisLayer = this.Layers{currentLayer}.LearnableParameters;
                layerIsLearning(currentLayer) = ~isempty(learnablesThisLayer) && any([ learnablesThisLayer.LearnRateFactor ]);
                numLearnables = numLearnables + numel(learnablesThisLayer);

                [layerOutputs{currentLayer}, memory{currentLayer}] = ...
                    iExecuteWithStagedGPUOOMRecovery( ...
                    @() this.Layers{currentLayer}.forward( ...
                    layerOutputs{currentLayer-1} ), 2, ...
                    recoveryStrategies, currentLayer );

                % 'In-place' ReLU. The input to a ReLU can be set equal to
                % its output without loss of accuracy (as long as the layer
                % before it doesn't need the input to the ReLU for
                % backprop). By making a copy (which will share the same
                % GPU memory) we can retrieve the memory used by that 
                % array.
                if isa(this.Layers{currentLayer}, 'nnet.internal.cnn.layer.ReLU') && ...
                   iLayerSupportsInPlaceReLU(this.Layers{currentLayer-1})
                    layerOutputs{currentLayer-1} = layerOutputs{currentLayer};
                end
                
                % Compute state information needed to update this layer if
                % this layer needs stateful training
                if needsStatefulTraining(currentLayer)
                    states{currentLayer} = computeState(this.Layers{currentLayer}, ...
                        layerOutputs{currentLayer-1}, layerOutputs{currentLayer}, ...
                        memory{currentLayer}, propagateState);
                end
                
                % Throw away activations that aren't going to be used
                % again. Backpropagation stops at the first learnable
                % layer.
                if ~any(layerIsLearning)
                    layerOutputs{currentLayer-1} = [];
                end
            end
            predictions = layerOutputs{end};

            % Compute the backward loss on the output layer
            dZ = iExecuteWithStagedGPUOOMRecovery( ...
                @() this.Layers{indexOutputLayer}.backwardLoss( ...
                layerOutputs{indexOutputLayer-1}, Y), 1, ...
                recoveryStrategies, indexOutputLayer );

            % Set up the backpropagation function, which calls backwards on
            % each layer and then discards the activations and memory when
            % they are no longer needed
            function gradientsThisLayer = efficientBackProp(currentLayer)
                % Compute backward loss
                learnablesThisLayer = this.Layers{currentLayer}.LearnableParameters;
                backwardArgs = { layerOutputs{currentLayer-1}, layerOutputs{currentLayer}, ...
                    dZ, memory{currentLayer} };
                if layerIsLearning(currentLayer)
                    [dX, gradientsThisLayer] = this.Layers{currentLayer}.backward( backwardArgs{:} );
                else
                    dX = this.Layers{currentLayer}.backward( backwardArgs{:} );
                    gradientsThisLayer = cell(size(learnablesThisLayer));
                end
                
                % Discard activations
                layerOutputs{currentLayer} = [];
                memory{currentLayer} = [];

                % Set output loss for next layer
                dZ = dX;
            end
            
            % To optimize away unnecessary backpropagation, determine
            % the earliest layer that needs its weight gradients computed
            earliestLearningLayer = find( layerIsLearning, 1, 'first' );
            
            % Propagate loss and gradients back through the network
            for el = indexOutputLayer-1:-1:earliestLearningLayer
                theseGradients = iExecuteWithStagedGPUOOMRecovery( ...
                    @() efficientBackProp(el), 1, recoveryStrategies, el );
                gradients = [theseGradients gradients]; %#ok<AGROW>
            end
            % Pad gradients to correct length
            gradients = [ cell(1, numLearnables-numel(gradients))  gradients ];
        end

        %%
        function net = finalizeNetwork(net, X)
            % finalizeNetwork
            
            % Work out how far through the network we need to propagate
            needsFinalize = cellfun(@(x) isa(x,'nnet.internal.cnn.layer.Finalizable'), net.Layers);
            lastLayer = find(needsFinalize, 1, 'last');
            assert(~isempty(lastLayer)); % Should never be called if no finalization required
            
            % Go forward through each layer, calling finalize if required.
            % First layer is input and never needs finalization.
            [Z, ~] = net.Layers{1}.forward( X );
            for currentLayer = 2:lastLayer
                % This layer's input is last layer's output
                X = Z;
                [Z, memory] = net.Layers{currentLayer}.forward(X);

                if needsFinalize(currentLayer)
                    net.Layers{currentLayer} = finalize(net.Layers{currentLayer}, X, Z, memory);
                end
            end

        end
        
        function loss = loss(this, predictions, response)
            % loss   Calculate the network loss
            loss = this.Layers{this.indexOutputLayer}.forwardLoss(predictions, response);
        end
        
        function this = updateLearnableParameters(this, deltas)
            % updateLearnableParameters   Update each learnable parameter
            % by subtracting a delta from it
            currentDelta = 1;
            for el = 1:this.indexOutputLayer
                for param = 1:numel(this.Layers{el}.LearnableParameters)
                    if ~isempty( deltas{currentDelta} )
                        this.Layers{el}.LearnableParameters(param).Value = this.Layers{el}.LearnableParameters(param).Value + deltas{currentDelta};
                    end
                    currentDelta = currentDelta + 1;
                end
            end
        end
        
        function this = updateNetworkState(this, states, statefulLayers)
            % updateNetworkState   Update network using state information
            % computed during gradient computation
            %
            % Inputs
            %   states                - cell array of state information
            %                           needed to update layer states after
            %                           gradient computation
            %   statefulLayers        - logical scalar for each layer
            %                           marking whether the layer needs
            %                           stateful training or not
            % Output
            %   this                  - network with updated state
            
            indexOutputLayer = this.indexOutputLayer();
            for currentLayer = 2:indexOutputLayer-1
                if statefulLayers(currentLayer)
                    this.Layers{currentLayer} = this.Layers{currentLayer}.updateState( states{currentLayer} );
                end
            end
        end
        
        function this = resetNetworkState(this, statefulLayers)
            % resetNetworkState   Reset the stateful layers of the network
            % to their initial states
            %
            % Inputs
            %   statefulLayers        - logical scalar for each layer
            %                           marking whether the layer needs
            %                           stateful training or not
            % Output
            %   this                  - network in initial state
            
            indexOutputLayer = this.indexOutputLayer();
            for currentLayer = 2:indexOutputLayer-1
                if statefulLayers(currentLayer)
                    initialState = this.Layers{currentLayer}.computeState([], [], [], false);
                    this.Layers{currentLayer} = this.Layers{currentLayer}.updateState( initialState );
                end
            end
        end
        
        function this = prepareNetworkForTraining(this, executionSettings)
            % prepareNetworkForTraining   Convert the network into a format
            % suitable for training
            for el = 1:this.indexOutputLayer
                this.Layers{el} = this.Layers{el}.prepareForTraining();
            end
            
            % Determine whether training should occur on host or GPU
            if ismember( executionSettings.executionEnvironment, {'gpu'} )
                % Don't move data if training in parallel, allow this to
                % happen as training progresses. This ensures we can
                % support clients without GPUs when the cluster has GPUs.
                delayMove = executionSettings.useParallel;
                this = this.setupNetworkForGPUTraining(delayMove);
            else
                this = this.setupNetworkForHostTraining();
            end
        end
        
        function this = prepareNetworkForPrediction(this)
            % prepareNetworkForPrediction   Convert the network into a 
            % format suitable for prediction
            for el = 1:this.indexOutputLayer
                this.Layers{el} = this.Layers{el}.prepareForPrediction();
            end
        end
        
        function this = setupNetworkForHostPrediction(this)
            % setupNetworkForHostPrediction   Setup the network to perform
            % prediction on the host
            for el = 1:this.indexOutputLayer
                this.Layers{el} = this.Layers{el}.setupForHostPrediction();
            end
        end
        
        function this = setupNetworkForGPUPrediction(this)
            % setupNetworkForGPUPrediction   Setup the network to perform
            % prediction on the GPU
            for el = 1:this.indexOutputLayer
                this.Layers{el} = this.Layers{el}.setupForGPUPrediction();
            end
        end
        
        function this = setupNetworkForHostTraining(this)
           % setupNetworkForHostTraining   Setup the network to train on
           % the host
           for el = 1:this.indexOutputLayer
              this.Layers{el} = this.Layers{el}.setupForHostTraining();
              this.Layers{el} = this.Layers{el}.moveToHost();
           end
        end
        
        function this = setupNetworkForGPUTraining(this, deferMove)
           % setupNetworkForGPUTraining   Setup the network to train on
           % the GPU. deferMove allows the actual move of data to the GPU
           % to be deferred to happen as training progresses instead of in
           % advance.
           for el = 1:this.indexOutputLayer
              this.Layers{el} = this.Layers{el}.setupForGPUTraining();
              if ~deferMove
                  this.Layers{el} = this.Layers{el}.moveToGPU();
              end
           end
        end

        function learnableParameters = get.LearnableParameters(this)
            learnableParameters = [];
            for el = 1:this.indexOutputLayer
                thisParam = this.Layers{el}.LearnableParameters;
                if ~isempty( thisParam )
                    learnableParameters = [learnableParameters thisParam]; %#ok<AGROW>
                end
            end
        end
    end
    
    methods (Access = private)
        function indexOutputLayer = indexOutputLayer(this)
            % indexOutputLayer    Return what number is the output layer
            indexOutputLayer = numel(this.Layers);
        end
    end
end

function tf = iInputLayerHasTransforms(layer)
% only input layers have transforms.
tf = isa(layer, 'nnet.internal.cnn.layer.ImageInput');
end

function cellsOnHost = iGatherGPUCell(cellsOnGpu)
cellsOnHost = cellfun(@gather, cellsOnGpu, 'UniformOutput', false);
end

function varargout = iExecuteWithStagedGPUOOMRecovery(varargin)
[varargout{1:nargout}] = nnet.internal.cnn.util.executeWithStagedGPUOOMRecovery(varargin{:});
end

function tf = iLayerSupportsInPlaceReLU(layerBeforeReLU)
% LSTM, BiLSTM and certain custom layers fo not support in-place ReLU
% because their output is needed for backpropagation.
tf = ~( isa(layerBeforeReLU, 'nnet.internal.cnn.layer.CustomLayer') || ...
    isa(layerBeforeReLU, 'nnet.internal.cnn.layer.LSTM') || ...
    isa(layerBeforeReLU, 'nnet.internal.cnn.layer.BiLSTM') );
end