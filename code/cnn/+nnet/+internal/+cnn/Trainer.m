classdef Trainer < handle
    % Trainer   Class for training a network
    
    %   Copyright 2015-2017 The MathWorks, Inc.
    
    properties(Access = protected)
        Options
        Schedule
        Precision
        Reporter
        ExecutionStrategy
        StopTrainingFlag
        InterruptException
    end
    
    methods
        function this = Trainer(opts, precision, reporter, executionSettings)
            % Trainer    Constructor for a network trainer
            %
            % opts - training options (nnet.cnn.TrainingOptionsSGDM)
            % precision - data precision
            this.Options = opts;
            scheduleArguments = iGetArgumentsForScheduleCreation(opts.LearnRateScheduleSettings);
            this.Schedule = nnet.internal.cnn.LearnRateScheduleFactory.create(scheduleArguments{:});
            this.Precision = precision;
            this.Reporter = reporter;
            % Declare execution strategy
            if ismember( executionSettings.executionEnvironment, {'gpu'} )
                this.ExecutionStrategy = nnet.internal.cnn.TrainerGPUStrategy;
            else
                this.ExecutionStrategy = nnet.internal.cnn.TrainerHostStrategy;
            end
            
            % Print execution environment if in verbose mode
            iPrintExecutionEnvironment(opts, executionSettings);
            
            % Register a listener to detect requests to terminate training
            addlistener( reporter, ...
                'TrainingInterruptEvent', @this.stopTrainingCallback);
        end
        
        function net = train(this, net, data)
            % train   Train a network
            %
            % Inputs
            %    net -- network to train
            %    data -- data encapsulated in a data dispatcher
            % Outputs
            %    net -- trained network
            reporter = this.Reporter;
            schedule = this.Schedule;
            prms = collectSettings(this, net);
            data.start();
            summary = nnet.internal.cnn.util.MiniBatchSummary(data, prms.maxEpochs);
            
            regularizer = iCreateRegularizer('l2',net.LearnableParameters,this.Precision,this.Options);
            
            solver = iCreateSolver(net.LearnableParameters,this.Precision,this.Options);
            
            trainingTimer = tic;
            
            reporter.start();
            iteration = 0;
            this.StopTrainingFlag = false;
            gradientThresholdOptions = struct('Method', this.Options.GradientThresholdMethod,...
                'Threshold', this.Options.GradientThreshold);
            gradThresholder = nnet.internal.cnn.GradientThresholder(gradientThresholdOptions);
            learnRate = initializeLearning(this);
            
            needsStatefulTraining = iNeedsStatefulTraining(data, net);

            if(this.Options.isGuided)  %IF Training Options Guided is True, Run GSGD
                rho = this.Options.Rho; %Number of Iterations to wait for assessment of consistent/inconsistent datasets before updating weights with guided approach
                RevisitBatchCount = this.Options.RevisitBatchNum; %Number of Previous Batches to Revisit
                verSetNum = this.Options.VerificationSetNum; %NUmber of batches to keep every epoch as verification data for calc of true error             
                
                prev_error = 10; %set initial true error to very large number                
                loopCount = 0;
                reVisit = false;
                avgBatchLosses = [];
                isGuided = false;

                % Start Epoch
                for epoch = 1:prms.maxEpochs
                    getVerificationData = true;
                    verSet_X = {};
                    verSet_response = {};
                    this.shuffle( data, prms.shuffleOption, epoch );
                    data.start();

                    % Start Training Iterations
                    while ~data.IsDone && ~this.StopTrainingFlag

                        %Set Verification Data at the beginning of epoch
                        if(getVerificationData)
                            for verCount=1:verSetNum
                                [X, response] = data.next();
                                verSet_X{verCount} = X; %#ok<AGROW>
                                verSet_response{verCount} = response; %#ok<AGROW>                            
                            end
                            getVerificationData = false;
                        end

                        if(~isGuided)
                            iteration = iteration + 1;
                            loopCount = loopCount +1;
                            [X, response] = data.next(); 
                            dataset_X{loopCount} = X; %#ok<AGROW>
                            dataset_response{loopCount} = response; %#ok<AGROW>


                            % Cast data to appropriate execution environment for
                            % training and apply transforms
                            X = this.ExecutionStrategy.environment(X);
                            response = this.ExecutionStrategy.environment(response);
                            X = apply(prms.transforms, X);
                            propagateState = iNeedsToPropagateState(data);
                            [gradients, predictions, states] = this.computeGradients(net, X, response, needsStatefulTraining, propagateState);
                            % Reuse the layers outputs to compute loss
                            miniBatchLoss = net.loss( predictions, response );
                            gradients = regularizer.regularizeGradients(gradients,net.LearnableParameters);
                            gradients = thresholdGradients(gradThresholder,gradients);
                            velocity = solver.calculateUpdate(gradients,learnRate);
                            net = net.updateLearnableParameters(velocity);
                            net = net.updateNetworkState(states, needsStatefulTraining);
                            elapsedTime = toc(trainingTimer);                        
                            summary.update(predictions, response, epoch, iteration, elapsedTime, miniBatchLoss, learnRate );
                            % It is important that computeIteration is called
                            % before reportIteration, so that the summary is
                            % correctly updated before being reported
                            reporter.computeIteration( summary, net );
                            reporter.reportIteration( summary );



                            % Get Verification Data Loss
                            verIDX = randperm(verSetNum,1);
                            X = verSet_X{verIDX};
                            response = verSet_response{verIDX};
                            % Cast data to appropriate execution environment for
                            % training and apply transforms
                            X = this.ExecutionStrategy.environment(X);
                            response = this.ExecutionStrategy.environment(response);
                            X = apply(prms.transforms, X);
                            propagateState = iNeedsToPropagateState(data);
                            [~, predictions, ~] = this.computeGradients(net, X, response, needsStatefulTraining, propagateState);
                            % Reuse the layers outputs to compute loss
                            verLoss = net.loss( predictions, response );
                            pos = 1;
                            if(verLoss<prev_error)
                                pos = 2;
                            end 

                            % Revist Previous Batches of Data and recalculate their
                            %losses only. WE DO NOT RE-UPDATE THE ENTIRE NETWORK WEIGHTS HERE.                    
                            if(reVisit)
                                if loopCount == 2
                                    loopEnd = loopCount;%In second loop, revisit only previous batch (batch 1)
                                else
                                    loopEnd = loopCount - (RevisitBatchCount-1);%In loops > 2, revisit previous 2 batches
                                end
                                currentBatchNumber = loopCount;
                                for k=loopCount:-1:loopEnd
                                    currentBatchNumber = currentBatchNumber - 1 ;
                                    X = dataset_X{currentBatchNumber};
                                    response = dataset_response{currentBatchNumber};

                                    % Cast data to appropriate execution environment for
                                    % training and apply transforms
                                    X = this.ExecutionStrategy.environment(X);
                                    response = this.ExecutionStrategy.environment(response);
                                    X = apply(prms.transforms, X);

                                    propagateState = iNeedsToPropagateState(data);
                                    [gradients, predictions, states] = this.computeGradients(net, X, response, needsStatefulTraining, propagateState);

                                    % Reuse the layers outputs to compute loss
                                    reVisitBatchLoss = net.loss( predictions, response ); 

                                    %previous batch was revisited and loss value is added into the array with previous batch losses                        
                                    prevBatchLosses = psi{currentBatchNumber};
                                    psi{currentBatchNumber} = [prevBatchLosses (power(-1,pos)) * (prev_error - reVisitBatchLoss)];  
                                end
                            end                       
                            %All batch error differences are collected into ?(psi).
                            current_batch_error = prev_error - verLoss;
                            psi{loopCount} = current_batch_error;
                            prev_error = verLoss;

                            reVisit = true;

                            % Check to see if its time for GSGD
                            if mod(loopCount,rho) == 0
                                isGuided = true;                       
                            end                       
                        else 
                            for k=1:loopCount
                                avgBatchLosses = [avgBatchLosses mean(psi{k})]; %#ok<AGROW>
                            end
                            [err,idx] = sort(avgBatchLosses,'descend');
                            minRepeat = min(rho/2, numel(avgBatchLosses));
                            for r=1:minRepeat
                                if(err(r)>0)
                                    guidedIdx = idx(r);
                                    X = dataset_X{guidedIdx};
                                    response = dataset_response{guidedIdx};
                                    X = this.ExecutionStrategy.environment(X);
                                    response = this.ExecutionStrategy.environment(response);
                                    X = apply(prms.transforms, X);

                                    propagateState = iNeedsToPropagateState(data);
                                    [gradients, predictions, states] = this.computeGradients(net, X, response, needsStatefulTraining, propagateState);

                                    % Reuse the layers outputs to compute loss
                                    miniBatchLoss = net.loss( predictions, response );

                                    gradients = regularizer.regularizeGradients(gradients,net.LearnableParameters);

                                    gradients = thresholdGradients(gradThresholder,gradients);

                                    velocity = solver.calculateUpdate(gradients,learnRate);

                                    net = net.updateLearnableParameters(velocity);

                                    net = net.updateNetworkState(states, needsStatefulTraining);

                                    elapsedTime = toc(trainingTimer);
% 
%                                     summary.update(predictions, response, epoch, iteration, elapsedTime, miniBatchLoss, learnRate );
%                                     % It is important that computeIteration is called
%                                     % before reportIteration, so that the summary is
%                                     % correctly updated before being reported
%                                     reporter.computeIteration( summary, net );
%                                     reporter.reportIteration( summary );


                                    % Get Verification Data Loss
                                    verIDX = randperm(verSetNum,1);
                                    X = verSet_X{verIDX};
                                    response = verSet_response{verIDX};
                                    % Cast data to appropriate execution environment for
                                    % training and apply transforms
                                    X = this.ExecutionStrategy.environment(X);
                                    response = this.ExecutionStrategy.environment(response);
                                    X = apply(prms.transforms, X);
                                    propagateState = iNeedsToPropagateState(data);
                                    [gradients, predictions, states] = this.computeGradients(net, X, response, needsStatefulTraining, propagateState);
                                    % Reuse the layers outputs to compute loss
                                    verLoss = net.loss( predictions, response );
                                    prev_error = verLoss;   
                                end
                            end 

                            avgBatchLosses = [];
                            psi = {};
                            dataset_X = {};
                            dataset_response = {};
                            loopCount = 0;  
                            reVisit = false;
                            isGuided = false; 
                        end
                    end
                    learnRate = schedule.update(learnRate, epoch);

                    reporter.reportEpoch( epoch, iteration, net );

                    % If an interrupt request has been made, break out of the
                    % epoch loop
                    if this.StopTrainingFlag
                        break;
                    end
                end
                reporter.computeFinish( summary, net );
                reporter.finish( summary );
            
            else %NOT GUIDED TRAINING
                for epoch = 1:prms.maxEpochs
                    this.shuffle( data, prms.shuffleOption, epoch );
                    data.start();
                    while ~data.IsDone && ~this.StopTrainingFlag
                        [X, response] = data.next();
                        % Cast data to appropriate execution environment for
                        % training and apply transforms
                        X = this.ExecutionStrategy.environment(X);
                        response = this.ExecutionStrategy.environment(response);
                        X = apply(prms.transforms, X);

                        propagateState = iNeedsToPropagateState(data);
                        [gradients, predictions, states] = this.computeGradients(net, X, response, needsStatefulTraining, propagateState);

                        % Reuse the layers outputs to compute loss
                        miniBatchLoss = net.loss( predictions, response );

                        gradients = regularizer.regularizeGradients(gradients,net.LearnableParameters);

                        gradients = thresholdGradients(gradThresholder,gradients);

                        velocity = solver.calculateUpdate(gradients,learnRate);

                        net = net.updateLearnableParameters(velocity);

                        net = net.updateNetworkState(states, needsStatefulTraining);

                        elapsedTime = toc(trainingTimer);

                        iteration = iteration + 1;
                        summary.update(predictions, response, epoch, iteration, elapsedTime, miniBatchLoss, learnRate );
                        % It is important that computeIteration is called
                        % before reportIteration, so that the summary is
                        % correctly updated before being reported
                        reporter.computeIteration( summary, net );
                        reporter.reportIteration( summary );
                    end
                    learnRate = schedule.update(learnRate, epoch);

                    reporter.reportEpoch( epoch, iteration, net );

                    % If an interrupt request has been made, break out of the
                    % epoch loop
                    if this.StopTrainingFlag
                        break;
                    end
                end
                reporter.computeFinish( summary, net );
                reporter.finish( summary );             
            end
        end
        
        function net = initializeNetworkNormalizations(this, net, data, precision, executionSettings, verbose)
            
            % Setup reporters
            this.Reporter.setup();
            
            % Always use 'truncateLast' as we want to process only the data we have.
            savedEndOfEpoch = data.EndOfEpoch;
            data.EndOfEpoch = 'truncateLast';
            
            networkInfo = nnet.internal.cnn.util.ComputeNetworkInfo(net);
            if networkInfo.ShouldImageNormalizationBeComputed
                if verbose
                    iPrintMessage('nnet_cnn:internal:cnn:Trainer:InitializingImageNormalization');
                end
                augmentations = iGetAugmentations(net);
                avgI = this.ExecutionStrategy.computeAverageImage(data, augmentations, executionSettings);
                net.Layers{1}.AverageImage = precision.cast(avgI);
            end
            
            data.EndOfEpoch = savedEndOfEpoch;
        end
        
        function net = finalizeNetwork(this, net, data)
            % Perform any finalization steps required by the layers
            
            % Always use 'truncateLast' as we want to process all the data we have.
            savedEndOfEpoch = data.EndOfEpoch;
            data.EndOfEpoch = 'truncateLast';
            
            % Call shared implementation
            net = this.doFinalize(net, data);
            
            data.EndOfEpoch = savedEndOfEpoch;
        end
    end

    methods(Access = protected)
        function stopTrainingCallback(this, ~, ~)
            % stopTraining  Callback triggered by interrupt events that
            % want to request training to stop
            this.StopTrainingFlag = true;
        end

        function settings = collectSettings(this, net)
            % collectSettings  Collect together fixed settings from the
            % Trainer and the data and put in the correct form.
            settings.maxEpochs = this.Options.MaxEpochs;
            settings.lossFunctionType = iGetLossFunctionType(net);
            settings.shuffleOption = this.Options.Shuffle;
            settings.transforms = [iGetAugmentations(net) iGetNormalization(net)];
        end

        function learnRate = initializeLearning(this)
            % initializeLearning  Set initial learning rate.
            learnRate = this.Precision.cast( this.Options.InitialLearnRate );
        end

        function [gradients, predictions, states] = computeGradients(~, net, X, Y, needsStatefulTraining, propagateState)
            % computeGradients   Compute the gradients of the network. This
            % function returns also the network output so that we will not
            % need to perform the forward propagation step again.
            [gradients, predictions, states] = net.computeGradientsForTraining(X, Y, needsStatefulTraining, propagateState);
        end

        function net = doFinalize(this, net, data)
            % Perform any finalization steps required by the layers
            needsFinalize = cellfun(@(x) isa(x,'nnet.internal.cnn.layer.Finalizable'), net.Layers);
            if any(needsFinalize)
                prms = collectSettings(this, net);
                % Do one final epoch
                data.start();
                while ~data.IsDone
                    X = data.next();
                    if ~isempty(X) % In the parallel case X can be empty
                        % Cast data to appropriate execution environment for
                        % training and apply transforms
                        X = this.ExecutionStrategy.environment(X);
                        X = apply(prms.transforms, X);
                        % Ask the network to finalize
                        net = finalizeNetwork(net, X);
                    end
                end
                
            end
        end
    end
    
    methods(Access = protected)
        function shuffle(~, data, shuffleOption, epoch)
            % shuffle   Shuffle the data as per training options
            if ~isequal(shuffleOption, 'never') && ...
                    ( epoch == 1 || isequal(shuffleOption, 'every-epoch') )
                data.shuffle();
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

function tf = iNeedsToPropagateState(data)
tf = isa(data,'nnet.internal.cnn.sequence.SequenceDispatcher') && data.IsNextMiniBatchSameObs;
end

function tf = iNeedsStatefulTraining(data,net)
needsUpdate = cellfun(@(x) isa(x,'nnet.internal.cnn.layer.Updatable'), net.Layers);
haveSequenceDispatcher = isa(data,'nnet.internal.cnn.sequence.SequenceDispatcher');
tf = needsUpdate & haveSequenceDispatcher;
end

function t = iGetLossFunctionType(net)
if isempty(net.Layers)
    t = 'nnet.internal.cnn.layer.NullLayer';
else
    t = class(net.Layers{end});
end
end

function n = iGetNormalization(net)
if isempty(net.Layers)
    n = nnet.internal.cnn.layer.ImageTransform.empty;
elseif isa(net.Layers{1},'nnet.internal.cnn.layer.SequenceInput')
    n = nnet.internal.cnn.layer.ImageTransform.empty;
else
    n = net.Layers{1}.Transforms;
end
end

function a = iGetAugmentations(net)
if isempty(net.Layers)
    a = nnet.internal.cnn.layer.ImageTransform.empty;
elseif isa(net.Layers{1},'nnet.internal.cnn.layer.SequenceInput')
    a = nnet.internal.cnn.layer.ImageTransform.empty;
else
    a = net.Layers{1}.TrainTransforms;
end
end

function scheduleArguments = iGetArgumentsForScheduleCreation(learnRateScheduleSettings)
scheduleArguments = struct2cell(learnRateScheduleSettings);
end

function iPrintMessage(messageID, varargin)
string = getString(message(messageID, varargin{:}));
fprintf( '%s\n', string );
end

function iPrintExecutionEnvironment(opts, executionSettings)
% Print execution environment if in 'auto' mode
if opts.Verbose
    if ismember(opts.ExecutionEnvironment, {'auto'})
        if ismember(executionSettings.executionEnvironment, {'cpu'})
            iPrintMessage( ...
                'nnet_cnn:internal:cnn:Trainer:TrainingInSerialOnCPU');
        elseif ismember(executionSettings.executionEnvironment, {'gpu'})
            iPrintMessage( ...
                'nnet_cnn:internal:cnn:Trainer:TrainingInSerialOnGPU');
        end
    elseif ismember(opts.ExecutionEnvironment, {'parallel'})
        if ismember(executionSettings.executionEnvironment, {'cpu'})
            iPrintMessage( ...
                'nnet_cnn:internal:cnn:Trainer:TrainingInParallelOnCPUs');
        elseif ismember(executionSettings.executionEnvironment, {'gpu'})
            iPrintMessage( ...
                'nnet_cnn:internal:cnn:Trainer:TrainingInParallelOnGPUs');
        end
    end
end
end