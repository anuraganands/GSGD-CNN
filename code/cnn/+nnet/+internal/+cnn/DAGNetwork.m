classdef DAGNetwork < nnet.internal.cnn.TrainableNetwork
    % DAGNetwork   Class for a directed acyclic graph network
    
    %   Copyright 2017-2018 The MathWorks, Inc.
    
    properties
        % Topologically sorted layers
        Layers
        
        % Topologically sorted connections
        Connections
    end
    
    properties
        % NumInputLayers   The number of input layers for this network
        %   The number of input layers for this network. This property is
        %   public because it is needed by the other DAGNetwork object.
        NumInputLayers
        
        % NumOutputLayers   The number of output layers for this network
        %   The number of output layers for this network. This property is
        %   public because it is needed by the other DAGNetwork object.
        NumOutputLayers
        
        % InputLayerIndices   The indices of the input layers
        %   The indices of the input layers.
        InputLayerIndices
        
        % OutputLayerIndices   The indices of the output layers
        %   The indices of the output layers.
        OutputLayerIndices
        
        % InputSizes   Sizes of the network inputs
        InputSizes
        
        % Outputsizes   Sizes of the network outputs
        OutputSizes
        
        % TopologicalOrder  Topological order of layers in the 
        % OriginalLayers array
        TopologicalOrder
    end
    
    properties(Access = private)
        % NumActivations   Number of activations
        %   The number of unique output activations in the network.
        NumActivations
        
        % ListOfBufferInputIndices   List of buffer input indices
        %   When we do forward propagation for a graph, we store
        %   activations in a linear buffer. This property is a list, with 
        %   one entry for each layer. Each entry is a vector of the indices
        %   in the linear buffer that are the inputs to this layer. For
        %   example, if the 4th entry stores the vector [1 2], then that
        %   means the 1st and 2nd entries of the linear buffer are the
        %   inputs to the 4th layer.
        ListOfBufferInputIndices
        
        % ListOfBufferOutputIndices   List of buffer output indices
        %   When we do forward propagation for a graph, we store
        %   activations in a linear buffer. This property is a list, with
        %   one entry for each layer. Each entry is a vector of the indices
        %   in the linear buffer that are outputs from this layer. For
        %   example, if the 2nd entry stores the vector [3 4], then that
        %   means the 3rd and 4th entries of the linear buffer are the
        %   outputs from the 2nd layer.
        ListOfBufferOutputIndices
        
        % ListOfBufferIndicesForClearingForward   List of buffer entries
        % that can be cleared as we move forward through the network
        %   When we do forward propagation for a graph, we store
        %   activations in a linear buffer. This property is a list, with
        %   one entry for each layer. Each entry is a vector of the indices
        %   in the linear buffer that can be cleared after evaluating a
        %   given layer in a forward pass. For example, if the 4th entry
        %   stores the vector [2 3], then that means the 2nd and 3rd 
        %   entries of the linear buffer can be cleared after evaluating
        %   the 4th layer.
        ListOfBufferIndicesForClearingForward
        
        % ListOfBufferIndicesForClearingBackward   List of buffer entries
        % that can be cleared as we move backward through the network
        %   When we do backward propagation for a graph, we are able to
        %   delete any stored activations as we move backwards through the
        %   graph. This property is a list, with one entry for each layer.
        %   Each entry is a vector of the indices in the linear buffer that
        %   can be cleared after evaluating a given layer in a backward
        %   pass. For example, if the 5th entry stores the vector [6 7],
        %   then that means the 6th and 7th entries of the linear buffer
        %   can be cleared after evaluating the backward pass of the 5th
        %   layer.
        ListOfBufferIndicesForClearingBackward
        
        % EdgeTable
        EdgeTable
        
        % Sizes   The output sizes for each activation
        Sizes
        
        % LayerOutputSizes  The output sizes for each layer
        LayerOutputSizes
    end
    
    properties(Dependent, Access = private)
        % NumLayers
        NumLayers
    end
    
    properties (Dependent, SetAccess = private)
        % LearnableParameters    Learnable parameters of the networks
        %                        (vector of nnet.internal.cnn.layer.LearnableParameter)
        LearnableParameters
        
        % LayerGraph    A layer graph
        %   This contains an internal layer graph with the most recent
        %   learnable parameters and is created using the Layers and
        %   Connections properties.
        LayerGraph
        
        % OriginalLayers  Layers in the original order
        OriginalLayers
        
        % OriginalConnections  Connections in the original order
        OriginalConnections
    end
    
    methods
        function learnableParameters = get.LearnableParameters(this)
            learnableParameters = [];
            for el = 1:this.NumLayers
                thisParam = this.Layers{el}.LearnableParameters;
                if ~isempty( thisParam )
                    learnableParameters = [learnableParameters thisParam]; %#ok<AGROW>
                end
            end
        end
        
        function layerGraph = get.LayerGraph(this)
            layerGraph = makeTrainedLayerGraph(this);
        end
        
        function originalLayers = get.OriginalLayers(this)
            originalLayers = nnet.internal.cnn.LayerGraph.sortedToOriginalLayers(this.Layers, this.TopologicalOrder);
        end
        
        function originalConnections = get.OriginalConnections(this)
            originalConnections = nnet.internal.cnn.LayerGraph.sortedToOriginalConnections(this.Connections, this.TopologicalOrder);
            originalConnections = sortrows(originalConnections);
        end
    end
    
    methods
        function val = get.NumLayers(this)
            val = numel(this.Layers);
        end
    end
    
    methods
        function this = DAGNetwork(sortedLayerGraph, topologicalOrder)
            %DAGNetwork - Create an internal DAGNetwork.
            %   this = DAGNetwork(sortedLayerGraph, topologicalOrder)
            %   creates an internal DAGNetwork. Input sortedLayerGraph is
            %   an internal LayerGraph containing a topologically sorted
            %   array of internal layers and input topologicalOrder is a
            %   vector representing the indices of the sorted internal
            %   layers in the original (unsorted) array of internal layers.
            
            % Get sorted internal layers based on topological order
            sortedGraph = getAugmentedDigraph(sortedLayerGraph);
            sortedInternalLayers = sortedLayerGraph.Layers;
            
            this.Layers = sortedInternalLayers;
            
            % Create an edgeTable with variables EndNodes and EndPorts.
            % Here's an example of what edgeTable should look like:
            %
            % edgeTable =
            %
            %   6x2 table
            %
            %     EndNodes      EndPorts
            %     ________    ____________
            %
            %     1    2      [1x2 double]
            %     2    3      [1x2 double]
            %     3    4      [1x2 double]
            %     4    5      [1x2 double]
            %     5    6      [1x2 double]
            %     6    7      [1x2 double]
            sourceLayer = findnode(sortedGraph,sortedGraph.Edges.EndNodes(:,1));
            targetLayer = findnode(sortedGraph,sortedGraph.Edges.EndNodes(:,2));
            endPorts = sortedGraph.Edges.AllEndPorts;
            edgeTable = table([sourceLayer, targetLayer],endPorts,'VariableNames',{'EndNodes','EndPorts'});
            this.EdgeTable = edgeTable;
            
            this.NumInputLayers = iCountInputLayers(sortedInternalLayers);
            this.NumOutputLayers = iCountOutputLayers(sortedInternalLayers);
            this.InputLayerIndices = iGetInputLayerIndices(sortedInternalLayers);
            this.OutputLayerIndices = iGetOutputLayerIndices(sortedInternalLayers);
            
            this = inferSizes(this);
            this.InputSizes = iGetInputSizes(this.LayerOutputSizes, ...
                this.InputLayerIndices);
            this.OutputSizes = iGetOutputSizes(this.LayerOutputSizes, ...
                this.OutputLayerIndices);
            
            this.NumActivations = iGetNumActivations( ...
                sortedInternalLayers, ...
                edgeTable);
            
            this.ListOfBufferOutputIndices = iGenerateListOfBufferOutputIndices( ...
                sortedInternalLayers, ...
                edgeTable);
            
            this.ListOfBufferInputIndices = iGenerateListOfBufferInputIndices( ...
                sortedInternalLayers, ...
                edgeTable, ...
                this.ListOfBufferOutputIndices);
            
            this.ListOfBufferIndicesForClearingForward = ...
                iGenerateListOfBufferIndicesForClearingForward( ...
                this.ListOfBufferInputIndices, ...
                this.ListOfBufferOutputIndices, ...
                this.OutputLayerIndices, ...
                this.NumActivations);
            
            this.ListOfBufferIndicesForClearingBackward = ...
                iGenerateListOfBufferIndicesForClearingBackward( ...
                this.ListOfBufferOutputIndices, ...
                this.OutputLayerIndices, ...
                this.NumActivations);
            
            % Save the internal connections. A layer graph with the most
            % recent values of learnable parameters can be accessed using
            % the LayerGraph property.
            this.Connections = iExternalToInternalConnections(this.EdgeTable);
            
            % Save the original layer indices.
            this.TopologicalOrder = topologicalOrder;
        end
        
        function [activationsBuffer, memoryBuffer, layerIsLearning] = forwardPropagationWithMemory(this, X)
            % Forward propagation used by training. Note, this version
            % retains activations and memory, but deletes any that won't be
            % needed for backpropagation.
            
            % Wrap X in cell if needed
            X = iWrapInCell(X);
            
            % Allocate space for the activations.
            activationsBuffer = cell(this.NumActivations,1);
            memoryBuffer = cell(this.NumActivations,1);
            
            % We can recover GPU memory by gathering the current
            % intermediate activations and memory cell arrays back to the
            % host.
            function gatherLayerOutputsAndMemory()
                activationsBuffer = iGatherGPUCell(activationsBuffer);
                memoryBuffer = iGatherGPUCell(memoryBuffer);
            end
            recoveryStrategies = {@gatherLayerOutputsAndMemory};

            layerIsLearning = false(this.NumLayers, 1);
            for i = 1:this.NumLayers
                % Mark whether this layer can learn, for backpropagation
                % optimisation
                learnablesThisLayer = this.Layers{i}.LearnableParameters;
                layerIsLearning(i) = ~isempty(learnablesThisLayer) && any([ learnablesThisLayer.LearnRateFactor ]);

                if isa(this.Layers{i},'nnet.internal.cnn.layer.ImageInput')
                        [~, currentInputLayer] = find(this.InputLayerIndices == i);
                        
                        [outputActivations, memory] = iExecuteWithStagedGPUOOMRecovery( ...
                            @() this.Layers{i}.forward( X{currentInputLayer} ), ...
                            2, recoveryStrategies, i );
                else
                        XForThisLayer = iGetTheseActivationsFromBuffer( ...
                            activationsBuffer, ...
                            this.ListOfBufferInputIndices{i});
                        
                        [outputActivations, memory] = iExecuteWithStagedGPUOOMRecovery( ...
                            @() this.Layers{i}.forward( XForThisLayer ), ...
                            2, recoveryStrategies, i );
                end
                
                activationsBuffer = iAssignActivationsToBuffer( ...
                    activationsBuffer, ...
                    this.ListOfBufferOutputIndices{i}, ...
                    outputActivations);
                
                memoryBuffer = iAssignMemoryToBuffer( ...
                    memoryBuffer, ...
                    this.ListOfBufferOutputIndices{i}, ...
                    memory);

                % Throw away data from layers that aren't going to be
                % visited on the backward pass
                if ~any(layerIsLearning) && i > 1
                    indicesToClear = this.ListOfBufferIndicesForClearingForward{i};
                    activationsBuffer = iClearActivationsFromBuffer( ...
                        activationsBuffer, indicesToClear );
                    memoryBuffer = iClearActivationsFromBuffer( ...
                        memoryBuffer, indicesToClear );
                end
            end
        end
        
        function Y = predict(this, X)
            
            % Wrap X in cell if needed
            X = iWrapInCell(X);
            
            % Apply any transforms
            X = this.applyTransformsForInputLayers(X);            

            % Allocate space for the activations.
            activationsBuffer = cell(this.NumActivations,1);
            
            % Loop over topologically sorted layers to perform forward
            % propagation. Clear memory when activations are no longer
            % needed.
            for i = 1:this.NumLayers
                if isa(this.Layers{i},'nnet.internal.cnn.layer.ImageInput')
                        [~, currentInputLayer] = find(this.InputLayerIndices == i);
                    
                        outputActivations = this.Layers{i}.predict(X{currentInputLayer});
                else
                        XForThisLayer = iGetTheseActivationsFromBuffer( ...
                            activationsBuffer, ...
                            this.ListOfBufferInputIndices{i});
                    
                        outputActivations = this.Layers{i}.predict(XForThisLayer);
                end
                
                activationsBuffer = iAssignActivationsToBuffer( ...
                    activationsBuffer, ...
                    this.ListOfBufferOutputIndices{i}, ...
                    outputActivations);
                
                activationsBuffer = iClearActivationsFromBuffer( ...
                    activationsBuffer, ...
                    this.ListOfBufferIndicesForClearingForward{i});
            end
            
            % Return activations corresponding to output layers.
            Y = { activationsBuffer{ ...
                    [this.ListOfBufferOutputIndices{this.OutputLayerIndices}] ...
                    } };
        end
        
        function Z = activations(this, X, layerIndex)
            
            % Wrap X in cell if needed
            X = iWrapInCell(X);
            
            % Apply transforms for input layers
            X = this.applyTransformsForInputLayers(X);
            
            % Allocate space for the activations.
            activationsBuffer = cell(this.NumActivations,1);
            
            % Loop over topologically sorted layers to perform forward
            % propagation. Clear memory when activations are no longer
            % needed.
            for i = 1:layerIndex
                if isa(this.Layers{i},'nnet.internal.cnn.layer.ImageInput')
                        [~, currentInputLayer] = find(this.InputLayerIndices == i);
                    
                        outputActivations = this.Layers{i}.predict(X{currentInputLayer});
                else
                        XForThisLayer = iGetTheseActivationsFromBuffer( ...
                            activationsBuffer, ...
                            this.ListOfBufferInputIndices{i});
                    
                        outputActivations = this.Layers{i}.predict(XForThisLayer);
                end
                
                activationsBuffer = iAssignActivationsToBuffer( ...
                    activationsBuffer, ...
                    this.ListOfBufferOutputIndices{i}, ...
                    outputActivations);
                
                activationsBuffer = iClearActivationsFromBuffer( ...
                    activationsBuffer, ...
                    this.ListOfBufferIndicesForClearingForward{i});
            end
            
            % Return the activations for the specified layer
            Z = iWrapInCell(outputActivations);
        end
        
        function [gradients, predictions, states] = computeGradientsForTraining( ...
                this, X, Y, needsStatefulTraining, propagateState)
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
            %                            stateful training or not. Note
            %                            that DAG does not support stateful
            %                            training, so each element of this
            %                            vector should be false.
            %   propagateState         - logical scalar marking whether
            %                            state needs to be propagated or
            %                            not. Note that DAG does not
            %                            support stateful training, so this
            %                            value should be false.
            %
            % Output
            %   gradients   - cell array of gradients with one element for
            %                 each learnable parameter array
            %   predictions - the output from the last layer, needs to be
            %                 preserved during training to report progress
            %   states      - cell array of state information needed to
            %                 update layer states after gradient
            %                 computation. Note that DAG does not support
            %                 stateful training, so this is always empty.
            
            % DAG currently does not support stateful training. Assert that
            % the user has not requested stateful training.
            assert(all(~needsStatefulTraining));
            assert(~propagateState);
            states = {};
            
            % Wrap X and Y in cell if needed
            X = iWrapInCell(X);
            Y = iWrapInCell(Y);
            
            % Do forward and get all activations
            [activationsBuffer, memoryBuffer, layerIsLearning] = this.forwardPropagationWithMemory(X);
            
            % Set up the backpropagation function, which calls backwards on
            % each layer and then discards the activations and memory when
            % they are no longer needed
            dLossdXBuffer = cell(this.NumActivations,1);
            function dLossdW = efficientBackProp(currentLayer)
                
                % Preparation
                bufferInputIndices = this.ListOfBufferInputIndices{currentLayer};
                bufferOutputIndices = this.ListOfBufferOutputIndices{currentLayer};
                learnablesThisLayer = this.Layers{currentLayer}.LearnableParameters;
                dLossdW = cell(size(learnablesThisLayer));

                % Output layers
                if isa(this.Layers{currentLayer}, 'nnet.internal.cnn.layer.OutputLayer')
                    % Perform backpropagation for an output layer
                    ZForThisLayer = iGetTheseActivationsFromBuffer( ...
                        activationsBuffer, bufferOutputIndices);
                    [~, currentInputLayer] = find(this.OutputLayerIndices == currentLayer);
                    TForThisLayer = Y{currentInputLayer};
                    
                    dLossdX = this.Layers{currentLayer}.backwardLoss( ...
                        ZForThisLayer, TForThisLayer);
                    
                    dLossdXBuffer = iIncrementActivationsInBuffer( ...
                        dLossdXBuffer, bufferInputIndices, dLossdX);
                    
                % Input layers
                elseif isa(this.Layers{currentLayer}, 'nnet.internal.cnn.layer.ImageInput')
                    % Do nothing
                    
                % Other layers
                else
                    % Perform backpropagation for some other kind of
                    % layer
                    XForThisLayer = iGetTheseActivationsFromBuffer( ...
                        activationsBuffer, bufferInputIndices);
                    ZForThisLayer = iGetTheseActivationsFromBuffer( ...
                        activationsBuffer, bufferOutputIndices);
                    dLossdZ = iGetTheseActivationsFromBuffer( ...
                        dLossdXBuffer, bufferOutputIndices);
                    memory = iGetTheseActivationsFromBuffer( ...
                        memoryBuffer, bufferOutputIndices);
                    
                    % Compute either all gradients or only the activations
                    % gradients depending on whether this layer is learning
                    backwardArgs = { XForThisLayer, ZForThisLayer, dLossdZ, memory };
                    if layerIsLearning(currentLayer)
                        [dLossdX, dLossdW] = this.Layers{currentLayer}.backward( backwardArgs{:} );
                    else
                        dLossdX = this.Layers{currentLayer}.backward( backwardArgs{:} );
                    end
                    
                    dLossdXBuffer = iIncrementActivationsInBuffer( ...
                        dLossdXBuffer, bufferInputIndices, dLossdX );
                end
                
                % Delete data that is no longer needed
                indicesToClear = this.ListOfBufferIndicesForClearingBackward{currentLayer};
                activationsBuffer = iClearActivationsFromBuffer( ...
                    activationsBuffer, indicesToClear );
                memoryBuffer = iClearActivationsFromBuffer( ...
                    memoryBuffer, indicesToClear );
                dLossdXBuffer = iClearActivationsFromBuffer( ...
                    dLossdXBuffer, indicesToClear );
            end

            % We can recover GPU memory by gathering the current
            % intermediate activations back to the host.
            function gatherActivations()
                activationsBuffer = iGatherGPUCell(activationsBuffer);
            end
            recoveryStrategies = {@gatherActivations};
            % 
            % We could also recover the memory and backward loss buffers
            function gatherBuffers()
                memoryBuffer = iGatherGPUCell(memoryBuffer);
                dLossdXBuffer = iGatherGPUCell(dLossdXBuffer);
            end
            recoveryStrategies = [ recoveryStrategies {@gatherBuffers} ];
            %
            % We could also return gradients on the host instead of the GPU
            gradients = {};
            function gatherGradients()
                gradients = iGatherGPUCell(gradients);
            end
            recoveryStrategies = [ recoveryStrategies {@gatherGradients} ];
            
            % To optimize away unnecessary backpropagation, determine
            % the earliest layer that needs its weight gradients computed
            earliestLearningLayer = find( layerIsLearning, 1, 'first' );

            % Propagate loss and gradient back through the network
            for i = this.NumLayers:-1:1
                if i >= earliestLearningLayer
                    theseGradients = iExecuteWithStagedGPUOOMRecovery( ...
                        @() efficientBackProp(i), ...
                        1, recoveryStrategies, i );
                else
                    % Pad output even if propagation has stopped
                    theseGradients = cell(1, numel(this.Layers{i}.LearnableParameters) );
                end
                gradients = [theseGradients gradients]; %#ok<AGROW>
            end

            % Predict
            predictions = cell(1, this.NumOutputLayers);
            for i = 1:this.NumOutputLayers
                outputLayerBufferIndex = this.ListOfBufferOutputIndices{this.OutputLayerIndices(i)};
                predictions{i} = activationsBuffer{outputLayerBufferIndex};
            end
            predictions = predictions{1};
        end
        
        function loss = loss(this, Y, T)
            % Wrap Y and T in cell if needed
            Y = iWrapInCell(Y);
            T = iWrapInCell(T);
            
            % loss   Calculate the network loss
            loss = [];
            for i = 1:this.NumOutputLayers
                loss = [loss this.Layers{this.OutputLayerIndices(i)}.forwardLoss(Y{i}, T{i})]; %#ok<AGROW>
            end
            loss = sum(loss);
        end
        
        function this = updateLearnableParameters(this, deltas)
            % updateLearnableParameters   Update each learnable parameter
            % by subtracting a delta from it
            currentDelta = 1;
            for el = 1:this.NumLayers
                for param = 1:numel(this.Layers{el}.LearnableParameters)
                    if ~isempty( deltas{currentDelta} )
                        this.Layers{el}.LearnableParameters(param).Value = this.Layers{el}.LearnableParameters(param).Value + deltas{currentDelta};
                    end
                    currentDelta = currentDelta + 1;
                end
            end
        end
        
        function this = updateNetworkState(this, ~, ~)
        end
        
        function this = initializeLearnableParameters(this, precision)
            % initializeLearnableParameters   Initialize the learnable
            % parameters of the network
            for el = 1:this.NumLayers
                this.Layers{el} = this.Layers{el}.initializeLearnableParameters(precision);
            end
        end
        
        function this = prepareNetworkForTraining(this, executionSettings)
            % prepareNetworkForTraining   Convert the network into a format
            % suitable for training
            for el = 1:this.NumLayers
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
            for el = 1:this.NumLayers
                this.Layers{el} = this.Layers{el}.prepareForPrediction();
            end
        end
        
        function this = setupNetworkForHostPrediction(this)
            % setupNetworkForHostPrediction   Setup the network to perform
            % prediction on the host
            for el = 1:this.NumLayers
                this.Layers{el} = this.Layers{el}.setupForHostPrediction();
            end
        end
        
        function this = setupNetworkForGPUPrediction(this)
            % setupNetworkForGPUPrediction   Setup the network to perform
            % prediction on the GPU
            for el = 1:this.NumLayers
                this.Layers{el} = this.Layers{el}.setupForGPUPrediction();
            end
        end
        
        function this = setupNetworkForHostTraining(this)
            % setupNetworkForHostTraining   Setup the network to train on
            % the host
            for el = 1:this.NumLayers
                this.Layers{el} = this.Layers{el}.setupForHostTraining();
                this.Layers{el} = this.Layers{el}.moveToHost();
            end
        end
        
        function this = setupNetworkForGPUTraining(this, deferMove)
           % setupNetworkForGPUTraining   Setup the network to train on
           % the GPU. deferMove allows the actual move of data to the GPU
           % to be deferred to happen as training progresses instead of in
           % advance.
           for el = 1:this.NumLayers
              this.Layers{el} = this.Layers{el}.setupForGPUTraining();
              if ~deferMove
                  this.Layers{el} = this.Layers{el}.moveToGPU();
              end
           end
        end
        
        function indices = namesToIndices(this, stringArray)
            % namesToIndices   Convert a string array of layer names into
            % layer indices
            numLayersToMatch = numel(stringArray);
            indices = zeros(numLayersToMatch,1);
            layerNames = nnet.internal.cnn.layer.Layer.getLayerNames(this.Layers);
            for i = 1:numLayersToMatch
                indices(i) = find(strcmp(stringArray(i), layerNames));
            end
        end
        
         function this = finalizeNetwork(this, X)
            % Wrap X in cell if needed
            X = iWrapInCell(X);
             
            % finalizeNetwork
            
            activationsBuffer = cell(this.NumActivations,1);
          
           % Allocate space for the activations.
            
            for i = 1:this.NumLayers
                
                layerType = class(this.Layers{i});
                switch layerType
                    case 'nnet.internal.cnn.layer.ImageInput'
                        [~, currentInputLayer] = find(this.InputLayerIndices == i);
                    
                        [Z, memory] = this.Layers{i}.forward(X{currentInputLayer});
                    otherwise
                        XForThisLayer = iGetTheseActivationsFromBuffer( ...
                            activationsBuffer, ...
                            this.ListOfBufferInputIndices{i});
                    
                        [Z, memory] = this.Layers{i}.forward(XForThisLayer);
                end
                
                activationsBuffer = iAssignActivationsToBuffer( ...
                    activationsBuffer, ...
                    this.ListOfBufferOutputIndices{i}, ...
                    Z);
                
               activationsBuffer = iClearActivationsFromBuffer( ...
                    activationsBuffer, ...
                    this.ListOfBufferIndicesForClearingForward{i});
                
                if  isa(this.Layers{i},'nnet.internal.cnn.layer.Finalizable')
                    this.Layers{i} = finalize(this.Layers{i}, XForThisLayer, Z, memory);
                end
                
                            
            end           

         end
        
         function this = inferSizes(this)
             % inferSizes   Infer layer output sizes
             
             sortedInternalLayers = this.Layers;
             edgeTable = this.EdgeTable;
             
             numActivations = iGetNumActivations( ...
                 sortedInternalLayers, ...
                 edgeTable);
             
             listOfBufferOutputIndices = iGenerateListOfBufferOutputIndices( ...
                 sortedInternalLayers, ...
                 edgeTable);
             
             listOfBufferInputIndices = iGenerateListOfBufferInputIndices( ...
                 sortedInternalLayers, ...
                 edgeTable, ...
                 listOfBufferOutputIndices);
             
             this.Sizes = cell(numActivations,1);
             numLayers = numel(sortedInternalLayers);
             this.LayerOutputSizes = cell(numLayers,1);
             
             for i = 1:numLayers
                 if isa(sortedInternalLayers{i}, 'nnet.internal.cnn.layer.ImageInput')
                     inputSizesForThisLayer = sortedInternalLayers{i}.InputSize;
                 else
                     inputSizesForThisLayer = iGetInputsFromBuffer( ...
                         this.Sizes, listOfBufferInputIndices{i});
                 end
                 
                 sortedInternalLayers{i} = iInferSize( ...
                     sortedInternalLayers{i}, ...
                     inputSizesForThisLayer, ...
                     i);
                 
                 outputSizesForThisLayer = sortedInternalLayers{i}.forwardPropagateSize( ...
                     inputSizesForThisLayer);
                 this.Sizes = iAssignOutputsToBuffer( ...
                     this.Sizes, listOfBufferOutputIndices{i}, outputSizesForThisLayer);
                 this.LayerOutputSizes{i} = outputSizesForThisLayer;
             end
         end
         
         function layerOutputSizes = inferOutputSizesGivenInputSizes(this, inputSizes)
             % inferOutputSizesGivenInputSizes   Infer layer output sizes
             % given new input sizes for input layers.
             %
             % Suppose this internal DAG network has N layers which have
             % been topologically sorted and numbered from 1 to N. Suppose
             % the network has M input layers and they appear in positions
             % i_1, i_2, ..., i_M in the topologically sorted list.
             %
             % inputSizes       - is a length M cell array specifying the
             %                    input sizes for layers i_1, i_2, ..., i_M
             %                    in that order.
             %
             % layerOutputSizes - is a length N cell array such that
             %                    layerOutputSizes{i} is the output size
             %                    for layer i. If layer i has multiple
             %                    outputs then layerOutputSizes{i} is a
             %                    cell array of output sizes for layer i.
             
             sortedInternalLayers = this.Layers;
             edgeTable = this.EdgeTable;
             
             numActivations = iGetNumActivations( ...
                 sortedInternalLayers, ...
                 edgeTable);
             
             listOfBufferOutputIndices = iGenerateListOfBufferOutputIndices( ...
                 sortedInternalLayers, ...
                 edgeTable);
             
             listOfBufferInputIndices = iGenerateListOfBufferInputIndices( ...
                 sortedInternalLayers, ...
                 edgeTable, ...
                 listOfBufferOutputIndices);
             
             sizes = cell(numActivations,1);
             numLayers = numel(sortedInternalLayers);
             layerOutputSizes = cell(numLayers,1);
             
             for i = 1:numLayers
                 if isa(sortedInternalLayers{i}, 'nnet.internal.cnn.layer.ImageInput')
                     % For an image input layer, forwardPropagateSize sets
                     % the output size equal to the InputSize property.
                     % Since we don't want that, we force the output size
                     % to be equal to the specified input size.
                     [~, currentInputLayer] = find(this.InputLayerIndices == i);
                     inputSizesForThisLayer = inputSizes{currentInputLayer};
                     outputSizesForThisLayer = inputSizesForThisLayer;
                 else
                     inputSizesForThisLayer = iGetInputsFromBuffer( ...
                         sizes, listOfBufferInputIndices{i});
                     outputSizesForThisLayer = sortedInternalLayers{i}.forwardPropagateSize( ...
                         inputSizesForThisLayer);
                 end
                 
                 sizes = iAssignOutputsToBuffer( ...
                     sizes, listOfBufferOutputIndices{i}, outputSizesForThisLayer);
                 layerOutputSizes{i} = outputSizesForThisLayer;
             end
         end
         
         function layerGraph = makeTrainedLayerGraph(this)
             % makeTrainedLayerGraph - makes an internal Layer graph
             % with most recent values of learnable parameters
             layerGraph = iMakeInternalLayerGraph(this.OriginalLayers, this.OriginalConnections);
         end
    end
    
    methods(Access = private)
        function X = applyTransformsForInputLayers(this, X)
            numInputLayers = numel(X);
            for i = 1:numInputLayers
                currentLayer = this.InputLayerIndices(i);
                X{i} = apply(this.Layers{currentLayer}.Transforms, X{i});
            end
        end
    end
end

function layerGraph = iMakeInternalLayerGraph(layers, connections)
layerGraph = nnet.internal.cnn.LayerGraph(layers, connections);
end

function internalConnections = iExternalToInternalConnections( externalConnections )
externalEndNodes = externalConnections.EndNodes;
externalEndPorts = externalConnections.EndPorts;
numEndPortsPerEndNodes = cellfun(@(x) size(x,1), externalEndPorts);
internalEndPorts = cell2mat(externalEndPorts);
internalEndNodes = [repelem(externalEndNodes(:,1),numEndPortsPerEndNodes), repelem(externalEndNodes(:,2),numEndPortsPerEndNodes)];
internalConnections = [internalEndNodes(:,1),internalEndPorts(:,1),internalEndNodes(:,2),internalEndPorts(:,2)];
end

function X = iWrapInCell(X)
if ~iscell(X)
    X = {X};
end
end

function numInputLayers = iCountInputLayers(internalLayers)
numInputLayers = 0;
for i = 1:numel(internalLayers)
    if( iIsAnInputLayer(internalLayers{i}) )
        numInputLayers = numInputLayers + 1;
    end
end
end

function numOutputLayers = iCountOutputLayers(internalLayers)
numOutputLayers = 0;
for i = 1:numel(internalLayers)
    if( iIsAnOutputLayer(internalLayers{i}) )
        numOutputLayers = numOutputLayers + 1;
    end
end
end

function inputLayerIndices = iGetInputLayerIndices(internalLayers)
numLayers = numel(internalLayers);
inputLayerIndices = cell(1,numLayers);
for i = 1:numLayers
    if(iIsAnInputLayer(internalLayers{i}))
        inputLayerIndices{i} = i;
    end
end
inputLayerIndices = cat(2,inputLayerIndices{:});
end

function outputLayerIndices = iGetOutputLayerIndices(internalLayers)
numLayers = numel(internalLayers);
outputLayerIndices = cell(1,numLayers);
for i = 1:numLayers
    if(iIsAnOutputLayer(internalLayers{i}))
        outputLayerIndices{i} = i;
    end
end
outputLayerIndices = cat(2, outputLayerIndices{:});
end

function inputSizes = iGetInputSizes(sizes, inputLayerIndices)
numInputLayers = numel(inputLayerIndices);
inputSizes = cell(1, numInputLayers);
for i = 1:numInputLayers
    currentLayer = inputLayerIndices(i);
    inputSizes{i} = sizes{currentLayer};
end
end

function outputSizes = iGetOutputSizes(sizes, outputLayerIndices)
numOutputLayers = numel(outputLayerIndices);
outputSizes = cell(1, numOutputLayers);
for i = 1:numOutputLayers
    currentLayer = outputLayerIndices(i);
    outputSizes{i} = sizes{currentLayer};
end
end

function tf = iIsAnInputLayer(internalLayer)
tf = isa(internalLayer,'nnet.internal.cnn.layer.ImageInput');
end

function tf = iIsAnOutputLayer(internalLayer)
tf = isa(internalLayer,'nnet.internal.cnn.layer.OutputLayer');
end

function numActivations = iGetNumActivations(internalLayers, edgeTable)
numActivations = 0;
for i = 1:numel(internalLayers)
    numOutputsForThisLayer = nnet.internal.cnn.layer.util.getNumOutputs(internalLayers{i});
    if(isnan(numOutputsForThisLayer))
        numOutputsForThisLayer = countNumberOfUniqueOutputsFromLayer(edgeTable, i);
    end
    numActivations = numActivations + numOutputsForThisLayer;
end
end

function listOfBufferOutputIndices = iGenerateListOfBufferOutputIndices(sortedInternalLayers, edgeTable)
numLayers = numel(sortedInternalLayers);
listOfBufferOutputIndices = cell(numLayers, 1);
offset = 0;
for i = 1:numLayers
    numOutputsForThisLayer = nnet.internal.cnn.layer.util.getNumOutputs( ...
        sortedInternalLayers{i});
    if(isnan(numOutputsForThisLayer))
        % This layer has a variable number of outputs, so we need to count them.
        numOutputsForThisLayer = countNumberOfUniqueOutputsFromLayer(edgeTable, i);
    end
    listOfBufferOutputIndices{i} = (1:numOutputsForThisLayer) + offset;
    offset = offset + numOutputsForThisLayer;
end
end

function numUniqueOutputs = countNumberOfUniqueOutputsFromLayer(edgeTable, layerIndex)
% Get the connections coming out of this layer
edgesOutOfLayerTable = edgeTable((edgeTable.EndNodes(:,1) == layerIndex), {'EndNodes','EndPorts'});

% Count the number of unique ports coming out of this layer
numUniqueOutputs = countNumberOfUniqueOutputPorts(edgesOutOfLayerTable);
end

function numUniqueOutputPorts = countNumberOfUniqueOutputPorts(edgesOutOfLayerTable)
portList = cell2mat(edgesOutOfLayerTable.EndPorts);
startPortList = portList(:,1);
numUniqueOutputPorts = numel(unique(startPortList));
end

function listOfBufferInputIndices = iGenerateListOfBufferInputIndices( ...
    sortedInternalLayers, edgeTable, listOfBufferOutputIndices)

numLayers = numel(sortedInternalLayers);
listOfBufferInputIndices = cell(numLayers,1);

for i = 1:numLayers
    % Get the connections feeding into this layer
    edgesIntoLayerTable = edgeTable((edgeTable.EndNodes(:,2) == i), {'EndNodes','EndPorts'});

    % Map inputs for each layer to indices in the activations buffer
    listOfBufferInputIndices{i} = iMapInputPortsToBufferIndices(edgesIntoLayerTable, listOfBufferOutputIndices);
end

end

function bufferIndices = iGenerateListOfBufferIndicesForClearingForward( ...
    listOfBufferInputIndices, ...
    listOfBufferOutputIndices, ...
    outputLayerIndices, ...
    numActivations)

numLayers = numel(listOfBufferInputIndices);
bufferIndices = cell(numLayers,1);
outputLayerBufferIndices = [ listOfBufferOutputIndices{outputLayerIndices} ];

% Initialize clearableIndices to include all of the indices except indices
% for output layers, which should never be cleared.
clearableIndices = setdiff(1:numActivations, outputLayerBufferIndices);

% We loop backards over the layers. At each layer, we can clear everything
% except inputs to subsequent layers, and outputs from the network.
for i = numLayers:-1:1
    bufferIndices{i} = clearableIndices;
    clearableIndices = setdiff(clearableIndices, listOfBufferInputIndices{i});
end

% The indices will be set correctly, but some of them will be cleared
% multiple times. Now we iterate forward through the layers and remove any
% duplicate indices.
clearedIndices = [];
for i = 1:numLayers
    oldBufferIndices = bufferIndices{i};
    bufferIndices{i} = setdiff(bufferIndices{i}, clearedIndices);
    clearedIndices = oldBufferIndices;
end

end

function bufferIndices = iGenerateListOfBufferIndicesForClearingBackward( ...
    listOfBufferOutputIndices, ...
    outputLayerIndices, ...
    numActivations)

numLayers = numel(listOfBufferOutputIndices);
bufferIndices = cell(numLayers,1);
outputLayerBufferIndices = [ listOfBufferOutputIndices{outputLayerIndices} ];

% Initialise clearableIndices to include all of the indices except indices
% for output layers, which should never be cleared.
clearableIndices = setdiff(1:numActivations, outputLayerBufferIndices);

% We loop forwards over the layers. At each layer, we can clear everything
% except outputs for previous layers, and outputs from the network.
for i = 1:numLayers
    bufferIndices{i} = clearableIndices;
    clearableIndices = setdiff(clearableIndices, listOfBufferOutputIndices{i});
end

% The indices will be set correctly, but some of them will be cleared
% multiple times. Now we iterate backward through the layers and remove any
% duplicate indices.
clearedIndices = [];
for i = numLayers:-1:1
    oldBufferIndices = bufferIndices{i};
    bufferIndices{i} = setdiff(bufferIndices{i}, clearedIndices);
    clearedIndices = oldBufferIndices;
end

end

function activationsBuffer = iClearActivationsFromBuffer(activationsBuffer, indicesToClear)
% Note that this works even if indicesToClear is empty
activationsBuffer(indicesToClear) = {[]}; 
end

function listOfBufferInputIndices = iMapInputPortsToBufferIndices(edgesIntoLayerTable, listOfBufferOutputIndices)

edgeMatrix = iConvertEdgeTableToMatrix(edgesIntoLayerTable);
edgeMatrix = iSortEdgeMatrixByInputPort(edgeMatrix);
layerIndexList = edgeMatrix(:,1);
outputPortList = edgeMatrix(:,3);

% Map to buffer
firstBufferOutputIndices = iGetFirstBufferOutputIndices(listOfBufferOutputIndices);
listOfBufferInputIndices = firstBufferOutputIndices(layerIndexList) + outputPortList - 1;
listOfBufferInputIndices = listOfBufferInputIndices';

end

function outputMatrix = iSortEdgeMatrixByInputPort(inputMatrix)
[~, sortedIndices] = sort(inputMatrix(:,4));
outputMatrix = inputMatrix(sortedIndices,:);
end

function edgeMatrix = iConvertEdgeTableToMatrix(edgeTable)
portMatrix = cell2mat(edgeTable.EndPorts);
numUniqueEdges = size(portMatrix,1);
edgeMatrix = zeros(numUniqueEdges, 4);
edgeMatrix(:,3:4) = portMatrix;
edgeMatrix(:,1:2) = iExpandLayerMatrix(edgeTable, numUniqueEdges);
end

function expandedLayerMatrix = iExpandLayerMatrix(edgeTable, numUniqueEdges)
expandedLayerMatrix = zeros(numUniqueEdges, 2);
numNonUniqueEdges = height(edgeTable);
startIndex = 1;
for i = 1:numNonUniqueEdges
    % Each non-unique edge may correspond to several unique edges, because 
    % we have multiple ports.
    portList = edgeTable.EndPorts{i};
    numPortEdges = size(portList,1);
    expandedEdgeMatrix = repmat(edgeTable.EndNodes(i,:), [numPortEdges 1]);
    stopIndex = startIndex + numPortEdges - 1;
    expandedLayerMatrix(startIndex:stopIndex,:) = expandedEdgeMatrix;
    startIndex = stopIndex + 1;
end
end

function firstBufferOutputIndices = iGetFirstBufferOutputIndices(listOfBufferOutputIndices)
numLayers = numel(listOfBufferOutputIndices);
firstBufferOutputIndices = zeros(numLayers, 1);
for i = 1:numLayers
    firstBufferOutputIndices(i) = listOfBufferOutputIndices{i}(1);
end
end

function XForThisLayer = iGetTheseActivationsFromBuffer(activationsBuffer, inputIndices)
XForThisLayer = activationsBuffer(inputIndices);
if(iscell(XForThisLayer) && (numel(XForThisLayer) == 1))
    XForThisLayer = XForThisLayer{1};
end
end

function memoryBuffer = iAssignMemoryToBuffer(...
    memoryBuffer, ...
    bufferIndices, ...
    memory)
% FYI Batch norm stores its memory as a cell. 
for i = 1:numel(bufferIndices)
    memoryBuffer{bufferIndices(i)} = memory;
end
end

function activationsBuffer = iAssignActivationsToBuffer( ...
    activationsBuffer, ...
    bufferIndices, ...
    activations)
if iscell(activations)
    activationsBuffer(bufferIndices) = activations;
else
    activationsBuffer{bufferIndices} = activations;
end
end

function activationsBuffer = iIncrementActivationsInBuffer(activationsBuffer, bufferIndices, activations)

numActivationsFromLayer = numel(bufferIndices);
if ~iscell(activations)
    if isempty(activationsBuffer{bufferIndices})
        activationsBuffer{bufferIndices} = activations;
    else
        activationsBuffer{bufferIndices} = activationsBuffer{bufferIndices} + activations;
    end
else
    for i = 1:numActivationsFromLayer
        if isempty(activationsBuffer{bufferIndices(i)})
            activationsBuffer{bufferIndices(i)} = activations{i};
        else
            activationsBuffer{bufferIndices(i)} = activationsBuffer{bufferIndices(i)}+ activations{i};
        end
    end
end
end

function internalLayer = iInferSize(internalLayer, inputSize, index)
if(~internalLayer.HasSizeDetermined)
    % Infer layer size if its size is not determined
    try
        internalLayer = internalLayer.inferSize(inputSize);
    catch e
        throwWrongLayerSizeException( e, index );
    end
else
    % Otherwise make sure the size of the layer is correct
    iAssertCorrectSize( internalLayer, index, inputSize );
end
end

function activationsBuffer = iAssignOutputsToBuffer( ...
    activationsBuffer, ...
    outputIndices, ...
    outputActivations)

numOutputsFromLayer = numel(outputIndices);
if ~iscell(outputActivations)
    activationsBuffer{outputIndices} = outputActivations;
else
    for i = 1:numOutputsFromLayer
        activationsBuffer{outputIndices(i)} = outputActivations{i}; 
    end
end
end

function iAssertCorrectSize( internalLayer, index, inputSize )
% iAssertCorrectSize   Check that layer size matches the input size,
% otherwise the architecture would be inconsistent.
if ~internalLayer.isValidInputSize( inputSize )
    exception = iCreateExceptionFromErrorID('nnet_cnn:inferParameters:WrongLayerSize', index);
    throwAsCaller(exception);
end
end

function throwWrongLayerSizeException(e, index)
% throwWrongLayerSizeException   Throws a getReshapeDims:notSameNumel exception as
% a WrongLayerSize exception
if (strcmp(e.identifier,'MATLAB:getReshapeDims:notSameNumel'))
    exception = iCreateExceptionFromErrorID('nnet_cnn:inferParameters:WrongLayerSize', index);
    throwAsCaller(exception)
else
    rethrow(e)
end
end

function exception = iCreateExceptionFromErrorID(errorID, varargin)
exception = MException(message(errorID, varargin{:}));
end

function XForThisLayer = iGetInputsFromBuffer(layerOutputs, inputIndices)
XForThisLayer = layerOutputs(inputIndices);
if(iscell(XForThisLayer) && (numel(XForThisLayer) == 1))
    XForThisLayer = XForThisLayer{1};
end
end

function cellOrArray = iGatherGPUCell(cellOrArray)
if iscell(cellOrArray)
    cellOrArray = cellfun(@iGatherGPUCell, cellOrArray, 'UniformOutput', false);
elseif isa(cellOrArray, 'gpuArray')
    cellOrArray = gather(cellOrArray);
end
end

function varargout = iExecuteWithStagedGPUOOMRecovery(varargin)
[varargout{1:nargout}] = nnet.internal.cnn.util.executeWithStagedGPUOOMRecovery(varargin{:});
end