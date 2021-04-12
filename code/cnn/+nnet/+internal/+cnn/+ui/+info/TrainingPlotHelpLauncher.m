classdef TrainingPlotHelpLauncher < nnet.internal.cnn.ui.info.HelpLauncher
    % TrainingPlotHelpLauncher   Opens help links from the training progress plot.
    
    %   Copyright 2017 The MathWorks, Inc
    
    methods
        function openLearnMoreHelpPage(~)
            helpview(fullfile(docroot, 'nnet','nnet.map'), 'trainingProgressPlot_learn_more');
        end
    end
end
