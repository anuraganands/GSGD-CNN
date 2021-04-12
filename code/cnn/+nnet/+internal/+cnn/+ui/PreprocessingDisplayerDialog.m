classdef PreprocessingDisplayerDialog < nnet.internal.cnn.ui.PreprocessingDisplayer
    % PreprocessingDisplayerDialog   Shows preprocessing information via a dialog
    
    %   Copyright 2017 The MathWorks, Inc.
    
    properties(Access = private)
        % DialogFactory   (nnet.internal.cnn.ui.DialogFactory)
        DialogFactory
        
        % PreprocessingDialog   (dialog) The dialog to display
        PreprocessingDialog = []
    end
    
    methods
        function this = PreprocessingDisplayerDialog(dialogFactory)
            this.DialogFactory = dialogFactory;
        end
        
        function displayPreprocessing(this, ~)
            if ~this.isDialogOpen()
                this.PreprocessingDialog = this.DialogFactory.createWaitDialog(...
                    iPreprocessingDialogMessage(), iPreprocessingDialogTitleMessage(), 'NNET_CNN_TRAININGPLOT_PREPROPRECESSINGDIALOG');
            end
        end
        
        function hidePreprocessing(this, ~)
            if this.isDialogOpen()
                delete(this.PreprocessingDialog);
                this.PreprocessingDialog = [];
            end
        end
        
        function delete(this)
            if this.isDialogOpen()
                delete(this.PreprocessingDialog); 
            end
        end
    end
    
    methods(Access = private)
        function tf = isDialogOpen(this)
            tf = ~isempty(this.PreprocessingDialog) && isvalid(this.PreprocessingDialog);
        end
    end
end

function m = iPreprocessingDialogMessage()
m = message('nnet_cnn:internal:cnn:ui:trainingplot:InitializingImageNormalization');
end

function m = iPreprocessingDialogTitleMessage()
m = message('nnet_cnn:internal:cnn:ui:trainingplot:InitializingImageNormalizationDialogTitle');
end
