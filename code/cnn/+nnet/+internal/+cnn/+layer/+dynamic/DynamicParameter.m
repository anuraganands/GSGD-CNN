classdef DynamicParameter
    % DynamicParameter   Interface for dynamic parameters

    %   Copyright 2017 The MathWorks, Inc

    properties(Abstract)
        % Value   The value of the dynamic parameter
        %   An array which can be a gpuArray or a host array.
        Value

        % Remember   Logical which is true when the parameter should be
        % remebered
        Remember
    end
end