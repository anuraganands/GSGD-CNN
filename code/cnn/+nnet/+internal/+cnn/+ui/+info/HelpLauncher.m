classdef(Abstract) HelpLauncher < handle
    % HelpLauncher   Opens various external help resources
    
    %   Copyright 2017 The MathWorks, Inc
    
    methods(Abstract)
        openLearnMoreHelpPage(~)
        % openLearnMoreHelpPage   Opens learn more link in doc.
    end
end
