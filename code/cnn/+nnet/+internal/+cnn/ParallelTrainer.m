classdef ParallelTrainer < nnet.internal.cnn.Trainer
    % ParallelTrainer   Class for training a network in parallel
    
    %   Copyright 2016-2017 The MathWorks, Inc.

    properties( Access = private )
        % InterruptStream  DataQueue created on the root worker that can be
        % used to send instructions to the pool to stop training
        InterruptStream
        
        % UseGpu  Whether to do optimized reductions on the GPU
        UseGpu
    end
    
    methods
        function this = ParallelTrainer(opts, precision, reporters, executionSettings)
            % ParallelTrainer    Constructor for a parallel trainer
            %
            % opts - training options (nnet.cnn.TrainingOptionsSGDM)
            % precision - data precision
            % reporters - reporters for feedback during training
            % executionSettings - training environment (eg host or GPU)
            
            % Wrap reporters with ParallelReporter which knows how to call
            % back to the client from the workers
            reporters = nnet.internal.cnn.util.ParallelReporter( reporters );
            
            % Construct superclass
            this@nnet.internal.cnn.Trainer(opts, precision, reporters, executionSettings);
            
            % Record execution environment
            this.UseGpu = ismember( executionSettings.executionEnvironment, {'gpu'} );
        end
        
        function net = train(this, net, data)
            % train   Train a network
            %
            % Inputs
            %    net -- network to train
            %    data -- data encapsulated in a data dispatcher
            % Outputs
            %    net -- trained network
            
            this.Reporter.start();
            
            % In order for SPMD to correctly understand how we wish to use
            % the Composite of data dispatchers stored within the
            % DistributedDataDispatcher, we must copy it out to a local
            % variable. We cannot access the containing dispatcher inside
            % SPMD.
            distributedData = data.DistributedData;
            
            % Determine which worker is going to have the output, and
            % create a DataQueue on that worker for the client to
            % communicate with the running trainer.
            [interruptStreamOnWorkers, this.InterruptStream] = ...
                iGetInterruptStream(distributedData);
            
            % To invoke our remote training function we need to open an
            % SPMD block. Calling SPMD is costly, so it must encompass the
            % entire algorithm.
            % In this case we cannot use the distributed dispatcher's
            % computeInParallel helper API because we need to pass a
            % Composite (the interruptStreamOnWorkers), which must be
            % declared explicitly in the same function as the SPMD block.
            trainFunc = @this.trainLocal; % Must create this outside SPMD
            spmd
                % The RemoteDispatchAdapter mixin provides the compute
                % function, which allows us to execute code treating the
                % distributed data like a normal dispatcher
                net = distributedData.compute( trainFunc, 1, net, distributedData, interruptStreamOnWorkers );
            end
            
            % net is returned as a Composite, expecting further processing
            % on the pool
        end
                
        function net = finalizeNetwork(this, net, data)
            % Always use 'truncateLast' as we want to process all the data we have.
            savedEndOfEpoch = data.EndOfEpoch;
            data.EndOfEpoch = 'truncateLast';

            % Accumulate
            % As for train method, we cannot use computeInParallel because
            % net is a Composite as well as data
            distributedData = data.DistributedData;
            finalizeFunc = @this.remoteFinalize;
            spmd
                net = distributedData.compute( finalizeFunc, 1, distributedData, net );
            end
            % net is returned as a Composite, expecting further processing
            % on the pool
            
            data.EndOfEpoch = savedEndOfEpoch;
        end
    end
    
    methods( Access = protected )
        
        function stopTrainingCallback(this, ~, eventData)
        % stopTraining  Overload to send cancellation request to workers
            send(this.InterruptStream, eventData);
        end
    end
    
    methods( Access = private )
        
        function net = trainLocal(this, net, data, interruptStream)
        % Training loop on a single worker
            trainingTimer = tic;
            prms = collectSettings(this, net);
            data.start();
            summary = nnet.internal.cnn.util.ParallelMiniBatchSummary(data, prms.maxEpochs);
            
            % Set up solve strategies
            regularizer = iCreateRegularizer('l2',net.LearnableParameters,this.Precision,this.Options);            
            solver = iCreateSolver(net.LearnableParameters,this.Precision,this.Options);
            gradientThresholdOptions = struct('Method', this.Options.GradientThresholdMethod,...
                'Threshold', this.Options.GradientThreshold);
            gradThresholder = nnet.internal.cnn.GradientThresholder(gradientThresholdOptions);
            
            % Training loop - outer is epochs, inner is iterations
            iteration = 0;
            this.StopTrainingFlag = false;
            learnRate = initializeLearning(this);
            %
            needsStatefulTraining = false(numel(net.Layers), 1);
            for epoch = 1:prms.maxEpochs
                this.shuffle( data, prms.shuffleOption, epoch );
                data.start();
                while this.continueThisEpoch(data, interruptStream)
                    [X, response] = data.next();
                    
                    % Allow for finished or empty datastores
                    if ~isempty(X)
                        subBatchSize = size(X, 4);
                        % Cast data to appropriate execution environment for
                        % training
                        X = this.ExecutionStrategy.environment(X);
                        response = this.ExecutionStrategy.environment(response);
                        X = apply(prms.transforms, X);
                        propagateState = false;
                        [gradients, predictions] = this.computeGradients(net, X, response, needsStatefulTraining, propagateState);
                        miniBatchLoss = net.loss( predictions, response );
                    else
                        % For finished or empty datastores, use zero
                        % gradients. This ensures these workers receive
                        % gradients even though they are not contributing.
                        subBatchSize = 0;
                        gradients = iExtractZeroGradientsFromLearnableParameters( net.LearnableParameters );
                        predictions = [];
                        miniBatchLoss = 0;
                    end
                    % Compute overall minibatch size across pool
                    miniBatchSize = gplus(subBatchSize);
                    
                    % Merge and normalize gradients between workers by
                    % summation
                    normalizationFactor = subBatchSize/miniBatchSize;
                    gradients = iMergeGradients( gradients, this.UseGpu, normalizationFactor );
                    
                    % Update weights - this happens entirely on the
                    % worker, but the results are the same on every
                    % worker thus resulting in the same networks.
                    gradients = regularizer.regularizeGradients(gradients,net.LearnableParameters);
                    gradients = thresholdGradients(gradThresholder,gradients);
                    velocity = solver.calculateUpdate(gradients,learnRate);
                    net = net.updateLearnableParameters(velocity);
                    
                    % Update and report state
                    iteration = iteration + 1;
                    elapsedTime = toc(trainingTimer);
                    summary.update( predictions, response, ...
                        epoch, iteration, elapsedTime, ...
                        miniBatchLoss, learnRate, prms.lossFunctionType );
                    this.Reporter.computeIteration( summary, net );
                    this.Reporter.reportIteration( summary );
                end
                
                learnRate = this.Schedule.update(learnRate, epoch);
                
                this.Reporter.reportEpoch( epoch, iteration, net );
                
                % If an interrupt request has been made, break out of the
                % epoch loop
                if this.StopTrainingFlag
                    break;
                end
            end  % End of epoch loop

            this.Reporter.computeFinish( summary, net );
            this.Reporter.finish( summary );
        end
        
        function continueEpoch = continueThisEpoch(this, data, interruptStream)
            % Tests when the epoch training loop should end based on the
            % data left in the datastores across the workers, and interrupt
            % requests. These two things are set together to keep the
            % communication to a minimum. Implemented as a method so that
            % the StopTrainingFlag property can be set, to cause the epoch
            % loop to be exited as well.
            [stopRequest, exception] = iReceivedInterrupt(interruptStream);
            stoppingConditions = ...
                struct('done', data.IsDone, 'stop', stopRequest, 'err', exception);
            stoppingConditions = gop(@iCatStruct, stoppingConditions);
            
            % If there were any errors, throw them collectively
            if ~isempty(stoppingConditions.err)
                throw( stoppingConditions.err(1) );
            end
            
            % We continue if any of the workers requested a stop or all
            % have finished their data
            isDone = all(stoppingConditions.done);
            this.StopTrainingFlag = any(stoppingConditions.stop);
            continueEpoch = ~isDone && ~this.StopTrainingFlag;
        end
        
        function net = remoteFinalize(this, distData, net)
        % Finalize algorithm running on the pool workers
        
            % Make sure data is in right place
            if this.UseGpu
                net = net.setupNetworkForGPUTraining(true);
            else
                net = net.setupNetworkForHostTraining();
            end
            net = this.doFinalize(net, distData);
            
            % Merge all finalizable layers, putting the result on
            % the first worker in the split communicator.
            for ii = 1:numel(net.Layers)
                if isa(net.Layers{ii},'nnet.internal.cnn.layer.Finalizable')
                    net.Layers{ii} = gop( @mergeFinalized, net.Layers{ii}, 1 );
                end
            end
            
            % Network returned should be ready for use on the client,
            % which may not have a GPU
            if labindex == distData.LowestRankComputeLabIndex
                net = net.setupNetworkForHostTraining();
            end
        end

    end
    
end

function [interruptStreamWorkers, interruptStreamClient] = ...
    iGetInterruptStream( distributedData )
% Determines which worker will hold the output network during training, and
% creates a DataQueue on that worker for the client to communicate with to
% control training.
spmd
    % The worker with the result will be the compute worker with the lowest
    % rank
    isWorkerActive = distributedData.IsComputeWorker;
    labIndexWithOutput = labindex;
    if ~isWorkerActive
        labIndexWithOutput = inf;
    end
    labIndexWithOutput = gop(@min, labIndexWithOutput);
    
    % Create a DataQueue on this worker and return a Composite
    interruptStreamWorkers = parallel.internal.pool.DataQueue.empty;
    if labindex == labIndexWithOutput
        interruptStreamWorkers = parallel.internal.pool.DataQueue;
    end
    
    % Return results to client
    interruptStreamClient = distributedutil.AutoTransfer( interruptStreamWorkers, labIndexWithOutput );
end

% Retrieve underlying data
interruptStreamClient = interruptStreamClient.Value;
end

function [stop, exception] = iReceivedInterrupt( interruptStream )
% Checks the data queue on the root worker to see if an interrupt
% is being requested
stop = false;
exception = [];
if ~isempty(interruptStream)
    [data, ok] = poll(interruptStream);
    if ok
        stop = true;
        if isa( data, 'nnet.internal.cnn.util.TrainingInterruptEventData' )
            exception = data.Exception;
        end
    end
end
end

function regularizer = iCreateRegularizer(name,learnableParameters,precision,regularizationOptions)
regularizer = nnet.internal.cnn.regularizer.RegularizerFactory.create(name,learnableParameters,precision,regularizationOptions);
end

function solver = iCreateSolver(learnableParameters,precision,trainingOptions)
solver = nnet.internal.cnn.solver.SolverFactory.create(learnableParameters,precision,trainingOptions);
end

function zeroGradients = iExtractZeroGradientsFromLearnableParameters(learnableParametersArray)
zeroGradients = cell(numel(learnableParametersArray),1);
for i = 1:numel(learnableParametersArray)
    thisParam = learnableParametersArray(i).Value;
    zeroGradients{i} = zeros( size(thisParam), 'like', thisParam );
end
end

function gradients = iMergeGradients( gradients, useGpu, normalizationFactor )
% Adds gradients from all workers together, minimising communication and
% taking account of empty gradient arrays. Normalization relative to the
% true mini-batch size is handled by normalizationFactor, which is the
% quotient of the sub-batch size and the mini-batch size. Multiplication by
% this factor "undoes" normalization of gradients by the sub-batch size
% performed by the local loss layer.

% Exit if there are no gradients (happens if nothing is learnable)
gradientSizes = cellfun(@numel, gradients, 'UniformOutput', true);
N = sum(gradientSizes);
if N == 0
    return;
end

% Concatenate all gradients so that we are communicating a single
% contiguous block of memory - this maximises the advantage of peer-to-peer
% communication for GPU devices
prototypeGradient = gradients{find( gradientSizes > 0, 1, 'first' )};
% Ensure on device, just in case gradients have been gathered to save
% memory
if useGpu
    % Empty index ensures we don't copy unnecessarily
    prototypeGradient = gpuArray( prototypeGradient([]) );
end
localGradVec = zeros(N, 1, 'like', prototypeGradient);
i = 1;
for g = 1:numel(gradients)
    n = gradientSizes(g);
    if n > 0
        localGradVec(i:(i+n-1)) = normalizationFactor.*gradients{g}(:);
        i = i + n;
    end
end

% MPI all-reduce collective - all workers will have same gradients
% Use peer-to-peer optimization for GPU if possible
if useGpu
    localGradVec = gplus(localGradVec, 'gpuArray');
else
    localGradVec = gplus(localGradVec);
end

% Put back into cell form
i = 1;
for g = 1:numel(gradients)
    n = gradientSizes(g);
    if n > 0
        % Use reshape rather than subsasgn in case LHS is host and RHS is gpu
        gradients{g} = reshape(localGradVec(i:(i+n-1)), size(gradients{g}));
        i = i + n;
    end
end
end

function s = iCatStruct(s1, s2)
% Concatenate the fields of two structures
s = s1;
f = fieldnames(s);
for i = 1:numel(f)
    fname = f{i};
    if ischar(s1.(fname))
        s.(fname) = {s1.(fname); s2.(fname)};
    else
        s.(fname) = [s1.(fname); s2.(fname)];
    end
end
end
