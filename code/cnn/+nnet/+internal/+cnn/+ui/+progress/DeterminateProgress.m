classdef(Abstract) DeterminateProgress < handle
    % DeterminateProgress   Interface which handles display of determinate progress of a process as well as the ability to stop that process.
    
    %   Copyright 2017 The MathWorks, Inc.
    
    properties(Abstract)
        % Parent   (graphics handle) The parent of MainPanel which holds
        % the progress bar.
        Parent 

        % Value   (double) The current value. The minimum is 0.
        Value 
        
        % Maximum   (positive double) The maximum value. 
        Maximum
    end
    
    properties(Abstract, SetAccess = private)
        % MainPanel   (uipanel) The main panel holding the progress bar
        MainPanel
        
        % PreferredHeight   (integer) The preferred height of MainPanel
        PreferredHeight
    end
    
    events
        % StopButtonClicked   Event fired when stop button was clicked
        StopButtonClicked
    end
end

