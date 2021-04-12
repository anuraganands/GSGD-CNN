classdef BiLSTM < nnet.internal.cnn.layer.Layer ...
        & nnet.internal.cnn.layer.Updatable
    % BiLSTM   Implementation of the bidirectional LSTM layer
    
    %   Copyright 2017 The MathWorks, Inc.
    
    properties
        % LearnableParameters   Learnable parameters for the layer
        % (Vector of nnet.internal.cnn.layer.learnable.PredictionLearnableParameter)
        LearnableParameters = nnet.internal.cnn.layer.learnable.PredictionLearnableParameter.empty();
        
        % DynamicParameters   Dynamic parameters for the layer
        % (Vector of nnet.internal.cnn.lauer.dynamic.TrainingDynamicParameter)
        DynamicParameters = nnet.internal.cnn.layer.dynamic.TrainingDynamicParameter.empty();
        
        % InitialCellState   Initial value of the cell state
        InitialCellState
        
        % Initial hidden state   Initial value of the hidden state
        InitialHiddenState

        % Name (char array)   A name for the layer
        Name
    end
    
    properties (Constant)
        % DefaultName   Default layer's name
        DefaultName = 'biLSTM'
    end
    
    properties (SetAccess = private)
        % InputNames   This layer has a single input
        InputNames = {'in'}
        
        % OutputNames   This layer has a single output
        OutputNames = {'out'}
        
        % InputSize (int) Size of the input vector
        InputSize
        
        % HiddenSize (int) Size of the hidden weights in the layer
        HiddenSize
        
        % ReturnSequence (logical) If true, output is a sequence. Otherwise
        % output is the last element in a sequence.
        ReturnSequence
        
        % RememberCellState (logical) If true, cell state is remembered
        RememberCellState
        
        % RememberHiddenState (logical) If true, hidden state is remembered
        RememberHiddenState
    end
    
    properties(Access = private)
        ExecutionStrategy
    end
    
    properties (Dependent)
        % Learnable Parameters (nnet.internal.cnn.layer.LearnableParameter)
        InputWeights
        RecurrentWeights
        Bias
        % Dynamic Parameters (nnet.internal.cnn.layer.DynamicParameter)
        CellState
        HiddenState
    end
    
    properties (Dependent, SetAccess = private)
        % HasSizeDetermined   Specifies if all size parameters are set
        %   If the input size has not been determined, then this will be
        %   set to false, otherwise it will be true
        HasSizeDetermined        
    end
    
    properties (Constant, Access = private)
        % InputWeightsIndex   Index of the Weights into the
        % LearnableParameters vector
        InputWeightsIndex = 1;
        
        % RecurentWeightsIndex   Index of the Recurrent Weights into the
        % LearnableParameters vector
        RecurrentWeightsIndex = 2;
        
        % BiasIndex   Index of the Bias into the LearnableParameters vector
        BiasIndex = 3;
        
        % CellStateIndex   Index of the cell state into the
        % DynamicParameters vector
        CellStateIndex = 1;
        
        % HiddenStateIndex   Index of the hidden state into the
        % DynamicParameters vector
        HiddenStateIndex = 2;
    end
    
    methods
        function this = BiLSTM(name, inputSize, hiddenSize, ...
                rememberCellState, rememberHiddenState, returnSequence)
            % BiLSTM   Constructor for an BiLSTM layer
            %
            %   Create an BiLSTM layer with the following
            %   compulsory parameters:
            %
            %   name                - Name for the layer [char array]
            %   inputSize           - Size of the input vector [int]
            %   hiddenSize          - Size of the hidden units [int]
            %   rememberCellState   - Remember the cell state [logical]
            %   rememberHiddenState - Remember the hidden state [logical]
            %   returnSequence      - Output as a sequence [logical]
           
            % Set layer name
            this.Name = name;
            
            % Set parameters
            this.InputSize = inputSize;
            this.HiddenSize = hiddenSize;
            this.RememberCellState = rememberCellState;
            this.RememberHiddenState = rememberHiddenState;
            this.ReturnSequence = returnSequence;
           
            % Set weights and bias to be LearnableParameter
            this.InputWeights = nnet.internal.cnn.layer.learnable.PredictionLearnableParameter();
            this.RecurrentWeights = nnet.internal.cnn.layer.learnable.PredictionLearnableParameter();
            this.Bias = nnet.internal.cnn.layer.learnable.PredictionLearnableParameter();
            
            % Set dynamic parameters
            this.CellState = nnet.internal.cnn.layer.dynamic.TrainingDynamicParameter();
            this.HiddenState = nnet.internal.cnn.layer.dynamic.TrainingDynamicParameter();
            
            % Initialize with host execution strategy
            this.ExecutionStrategy = this.getHostStrategy();
        end
        
        function [Z, memory] = forward( this, X )
            % forward   Forward input data through the layer at training
            % time and output the result
            [Z, memory] = this.ExecutionStrategy.forward( ...
                X, this.InputWeights.Value, this.RecurrentWeights.Value, ...
                this.Bias.Value, this.CellState.Value, this.HiddenState.Value);
        end
        
        function Z = predict( this, X )
            % predict   Forward input data through the layer at prediction
            % time and output the result
            Z = this.ExecutionStrategy.forward( ...
                X, this.InputWeights.Value, this.RecurrentWeights.Value, ...
                this.Bias.Value, this.CellState.Value, this.HiddenState.Value);
        end
        
        function [dX, dW] = backward( this, X, Z, dZ, memory, ~ )
            % backward    Back propagate the derivative of the loss function
            % through the layer
            [dX, dIW, dRW, dB] = this.ExecutionStrategy.backward(  ...
                X, this.InputWeights.Value, this.RecurrentWeights.Value, ...
                this.Bias.Value, this.CellState.Value, this.HiddenState.Value, ...
                Z, memory, dZ);
            
            dW{this.InputWeightsIndex} = dIW;
            dW{this.RecurrentWeightsIndex} = dRW;
            dW{this.BiasIndex} = dB;
        end
        
        function this = inferSize(this, inputSize)
            if ~this.HasSizeDetermined
                this.InputSize = inputSize;
            end
        end
        
        function outputSize = forwardPropagateSize(this, ~)
            outputSize = 2*this.HiddenSize;
        end
        
        function tf = isValidInputSize(this, inputSize)
            % isValidInputSize   Check if the layer can accept an input of
            % a certain size
            tf = ( ~this.HasSizeDetermined && isscalar(inputSize) ) ...
                || isequal(this.InputSize, inputSize);
        end
        
        function this = initializeLearnableParameters(this, precision)
            % initializeLearnableParameters    Initialize learnable
            % parameters using their initializer
            
            if isempty(this.InputWeights.Value)
                % Initialize only if it is empty
                inputSize = [8*this.HiddenSize, this.InputSize];
                this.InputWeights.Value = iInitializeGaussian( inputSize, precision );
            else
                % Cast to desired precision
                this.InputWeights.Value = precision.cast(this.InputWeights.Value);
            end
            
            if isempty(this.RecurrentWeights.Value)
                % Initialize only if it is empty
                hiddenSize = [8*this.HiddenSize, this.HiddenSize];
                this.RecurrentWeights.Value = iInitializeGaussian( hiddenSize, precision );
            else
                % Cast to desired precision
                this.RecurrentWeights.Value = precision.cast(this.RecurrentWeights.Value);
            end
            
            if isempty(this.Bias.Value)
                % Initialize only if it is empty
                this.Bias.Value = iInitializeBias( this.HiddenSize, precision );
            else
                % Cast to desired precision
                this.Bias.Value = precision.cast(this.Bias.Value);
            end
        end
        
        function this = initializeDynamicParameters(this, precision)
           % initializeDynamicParameters   Initialize dynamic parameters
           
           % Cell state
           if isempty(this.InitialCellState)
               parameterSize = [2*this.HiddenSize 1];
               this.InitialCellState = iInitializeConstant(parameterSize, precision);
           else
               this.InitialCellState = precision.cast(this.InitialCellState);
           end
           % Set the running cell state
           this.CellState.Value = this.InitialCellState;
           this.CellState.Remember = this.RememberCellState;
           
           % Hidden units
           if isempty(this.InitialHiddenState)
               parameterSize = [2*this.HiddenSize 1];
               this.InitialHiddenState = iInitializeConstant(parameterSize, precision);
           else
               this.InitialHiddenState = precision.cast(this.InitialHiddenState);
           end
           % Set the running hidden state
           this.HiddenState.Value = this.InitialHiddenState;
           this.HiddenState.Remember = this.RememberHiddenState;
        end
        
        function state = computeState(this, ~, ~, memory, propagateState)
            % state{1} - store cell state
            % state{2} - store hidden state
            state = cell(2,1);
            if propagateState
                % Here we take the final states of the previous forward
                % pass for the new forward-sequence states. This allows us
                % to propagate the forward-sequence states between
                % consecutive batches.
                CfS = memory.CellState(1:this.HiddenSize, :, end);
                HfS = memory.HiddenState(1:this.HiddenSize, :, end);
                % Here we take the initial states to be the new
                % backward-sequence states. This means that the
                % backward-sequence states are not propagated across
                % mini-batches
                N = size( CfS, 2 );
                Cb0 = repmat( this.InitialCellState((1+this.HiddenSize:end), 1), [1 N] );
                Hb0 = repmat( this.InitialHiddenState((1+this.HiddenSize:end), 1), [1 N] );
                state{1} = [CfS; Cb0];
                state{2} = [HfS; Hb0];
            else
                state{1} = this.InitialCellState;
                state{2} = this.InitialHiddenState;
            end
        end
        
        function this = updateState(this, state)
            % Update the cell state
            if this.DynamicParameters(this.CellStateIndex).Remember
                this.DynamicParameters(this.CellStateIndex).Value = state{1};
            end
            
            % Update the hidden state
            if this.DynamicParameters(this.HiddenStateIndex).Remember
                this.DynamicParameters(this.HiddenStateIndex).Value = state{2};
            end
        end        
        
        function this = prepareForTraining(this)
            % prepareForTraining   Prepare this layer for training
            %   Before this layer can be used for training, we need to
            %   convert the learnable parameters to use the class
            %   TrainingLearnableParameter.
            this.LearnableParameters = nnet.internal.cnn.layer.learnable.convert2training(this.LearnableParameters);
        end
        
        function this = prepareForPrediction(this)
            % prepareForPrediction   Prepare this layer for prediction
            %   Before this layer can be used for prediction, we need to
            %   convert the learnable parameters to use the class
            %   PredictionLearnableParameter.
            this.LearnableParameters = nnet.internal.cnn.layer.learnable.convert2prediction(this.LearnableParameters);
        end
        
        function this = setupForHostPrediction(this)        
            this.ExecutionStrategy = this.getHostStrategy();
            this.LearnableParameters(1).UseGPU = false;
            this.LearnableParameters(2).UseGPU = false;
            this.LearnableParameters(3).UseGPU = false;
        end
        
        function this = setupForGPUPrediction(this)
            this.ExecutionStrategy = this.getGPUStrategy();
            this.LearnableParameters(1).UseGPU = true;
            this.LearnableParameters(2).UseGPU = true;
            this.LearnableParameters(3).UseGPU = true;
        end
        
        function this = setupForHostTraining(this)
            this.ExecutionStrategy = this.getHostStrategy();
        end
        
        function this = setupForGPUTraining(this)
            this.ExecutionStrategy = this.getGPUStrategy();
        end
        
        % Setter and getter for InputWeights, RecurrentWeights and Bias
        % These make easier to address into the vector of LearnableParameters
        % giving a name to each index of the vector
        function weights = get.InputWeights(this)
            weights = this.LearnableParameters(this.InputWeightsIndex);
        end
        
        function this = set.InputWeights(this, weights)
            this.LearnableParameters(this.InputWeightsIndex) = weights;
        end
        
        function weights = get.RecurrentWeights(this)
            weights = this.LearnableParameters(this.RecurrentWeightsIndex);
        end
        
        function this = set.RecurrentWeights(this, weights)
            this.LearnableParameters(this.RecurrentWeightsIndex) = weights;
        end
        
        function bias = get.Bias(this)
            bias = this.LearnableParameters(this.BiasIndex);
        end
        
        function this = set.Bias(this, bias)
            this.LearnableParameters(this.BiasIndex) = bias;
        end
        
        % Setter and getter for CellState and HiddenState
        function state = get.CellState(this)
            state = this.DynamicParameters(this.CellStateIndex);
        end
        
        function this = set.CellState(this, state)
            this.DynamicParameters(this.CellStateIndex) = state;
        end
        
        function state = get.HiddenState(this)
            state = this.DynamicParameters(this.HiddenStateIndex);
        end
        
        function this = set.HiddenState(this, state)
            this.DynamicParameters(this.HiddenStateIndex) = state;
        end
        
        % Getter for HasSizeDetermined
        function tf = get.HasSizeDetermined(this)
            tf = ~isempty( this.InputSize );
        end 
    end
    
    methods(Access = private)
        function executionStrategy = getHostStrategy(this)
            if this.ReturnSequence
                executionStrategy = nnet.internal.cnn.layer.util.BiLSTMHostStrategy();
            else
                executionStrategy = nnet.internal.cnn.layer.util.BiLSTMHostReturnLastStrategy();
            end
        end
        
        function executionStrategy = getGPUStrategy(this)
            if this.ReturnSequence
                executionStrategy = nnet.internal.cnn.layer.util.BiLSTMGPUStrategy();
            else
                executionStrategy = nnet.internal.cnn.layer.util.BiLSTMGPUReturnLastStrategy();
            end
        end
    end
end

function parameter = iInitializeGaussian(parameterSize, precision)
parameter = precision.cast( iNormRnd(0, 0.01, parameterSize) );
end

function parameter = iInitializeConstant(parameterSize, precision)
parameter = precision.cast( zeros(parameterSize) );
end

function parameter = iInitializeBias(hiddenSize, precision)
% Initialize forget gate bias to unity, and other biases to zero. See
% http://jmlr.org/proceedings/papers/v37/jozefowicz15.pdf

% Get LSTM gate indices
[zInd, iInd, fInd, oInd] = nnet.internal.cnn.util.gateIndices(hiddenSize);
% Get forward/backward sequence weight indices
[forwardInd, backwardInd] = nnet.internal.cnn.util.forwardBackwardSequenceIndices(hiddenSize);
% Order weight indices as per LSTM gates
forwardInd = forwardInd([zInd iInd fInd oInd]);
backwardInd = backwardInd([zInd iInd fInd oInd]);
% Create bias vectors with forget-gate ones and zeros elsewhere
bias = [zeros(2*hiddenSize, 1); ones(hiddenSize, 1); zeros(hiddenSize, 1)];
% Assign and cast layer bias vector
parameter(forwardInd, :) = precision.cast( bias );
parameter(backwardInd, :) = precision.cast( bias );
end

function out = iNormRnd(mu, sigma, outputSize)
% iNormRnd  Returns an array of size 'outputSize' chosen from a
% normal distribution with mean 'mu' and standard deviation 'sigma'
out = randn(outputSize) .* sigma + mu;
end