classdef SolverFactory
    % SolverFactory   Class for creating solvers.
    
    %   Copyright 2017 The MathWorks, Inc.
    
    methods(Static)
        function solver = create(learnableParameters,precision,solverOptions)
            % solver = create(learnableParameters,precision,solverOptions)
            % creates a Solver object for optimizing parameters in
            % learnableParameters using floating point precision specified
            % in precision and solver options specified in solverOptions.
            % The class of inputs is as follows:
            %
            %    learnableParameters - an array of objects of type nnet.internal.cnn.layer.learnable.LearnableParameter
            %    precision           - an object of type nnet.internal.cnn.util.Precision
            %    solverOptions       - an object of a subclass of
            %    nnet.cnn.TrainingOptions 
            
            if iIsSolverSGDM(solverOptions)
                solver = nnet.internal.cnn.solver.SolverSGDM(learnableParameters,precision,solverOptions);
            elseif iIsSolverADAM(solverOptions)
                solver = nnet.internal.cnn.solver.SolverADAM(learnableParameters,precision,solverOptions);
            elseif iIsSolverTESTGD(solverOptions)
                solver = nnet.internal.cnn.solver.SolverTESTGD(learnableParameters,precision,solverOptions);
            elseif iIsSolverRMSProp(solverOptions)
                solver = nnet.internal.cnn.solver.SolverRMSProp(learnableParameters,precision,solverOptions);
            else
                error('Unsupported solver.');
            end
        end
    end
end

function tf = iIsSolverSGDM(solverOptions)
tf = isa(solverOptions,'nnet.cnn.TrainingOptionsSGDM');
end

function tf = iIsSolverTESTGD(solverOptions)
tf = isa(solverOptions,'nnet.cnn.TrainingOptionsTESTGD');
end

function tf = iIsSolverADAM(solverOptions)
tf = isa(solverOptions,'nnet.cnn.TrainingOptionsADAM');
end

function tf = iIsSolverRMSProp(solverOptions)
tf = isa(solverOptions,'nnet.cnn.TrainingOptionsRMSProp');
end