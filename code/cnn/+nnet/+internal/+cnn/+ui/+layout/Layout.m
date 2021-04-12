classdef(Abstract) Layout < handle
    % Layout   Interface for laying out objects in a panel.
    
    %   Copyright 2017 The MathWorks, Inc.
    
    properties(Abstract, SetAccess = private)
        % Parent   (graphics handle) The parent component
        Parent
        
        % MainPanel   (matlab.ui.container.Panel)
        % The main panel where all the laid out children will reside
        MainPanel
    end
    
    methods(Abstract)
        add(this, child, weight)
        % add   Add a child component to the MainPanel (with some 'weight')
    end
    
end

