classdef IsOfSameTypeAs < matlab.unittest.constraints.Constraint
    % IsOfSameTypeAs   Constraint to verify that two variables have the
    % same type
    
    %   Copyright 2017 The MathWorks, Inc.
    
    properties(SetAccess=immutable)
        ExpectedType
        ExpectedUnderlyingType = []
    end
    
    methods
        
        function constraint = IsOfSameTypeAs(value)
            constraint.ExpectedType = class(value);
            if isa( value, 'gpuArray')
                constraint.ExpectedUnderlyingType = classUnderlying(value);
            end
        end
        
        function bool = satisfiedBy(constraint, actual)
            bool = isSameMainType(constraint,actual) && isSameUnderlyingType(constraint,actual);
        end
        
        function diag = getDiagnosticFor(constraint, actual)
            import matlab.unittest.diagnostics.StringDiagnostic
            
            bool = constraint.satisfiedBy(actual);
            if bool
                diag = '';
            elseif ~isSameMainType(constraint,actual)
                diag = StringDiagnostic( ...
                    getString( message( 'nnet_cnn:nnet:checklayer:constraints:IsOfSameTypeAs:DifferentTypes', ...
                    class(actual), ...
                    constraint.ExpectedType ) ) );
            elseif ~isSameUnderlyingType(constraint,actual)
                diag = StringDiagnostic( ...
                    getString( message( 'nnet_cnn:nnet:checklayer:constraints:IsOfSameTypeAs:DifferentUnderlyingTypes', ...
                    classUnderlying(actual), ...
                    constraint.ExpectedUnderlyingType ) ) );
            end
        end
    end
    
    methods(Access=private)
        
        function bool = isSameMainType(constraint,actual)
            bool = isequal( class(actual), constraint.ExpectedType );
        end
        
        function bool = isSameUnderlyingType(constraint,actual)
            bool = isempty(constraint.ExpectedUnderlyingType) ...
                || isequal( classUnderlying(actual), constraint.ExpectedUnderlyingType );
        end
        
    end
end
