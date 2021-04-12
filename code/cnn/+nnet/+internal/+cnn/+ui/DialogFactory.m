classdef DialogFactory < handle
    % DialogFactory   Factory for creating dialogs
    
    %   Copyright 2017 The MathWorks, Inc.
    
    methods
        function h = createWaitDialog(~, message, titleMessage, tag)
            h = dialog( ...
                'Name', titleMessage.getString(), ...
                'Units', 'points', ...
                'Tag', tag, ...
                'WindowStyle', 'normal');
            
            messageOffset = 14;
            figureWidth = 250;
            figureHeight = 50;
            messageTextWidth = figureWidth - 2*messageOffset;
            messageTextHeight = figureHeight - 2*messageOffset;
            messagePosition = [messageOffset messageOffset messageTextWidth messageTextHeight];
            uicontrol(h, ...
                'Style', 'text', ...
                'Units', 'points', ...
                'Position', messagePosition, ...
                'String', message.getString(), ...
                'Tag', 'NNET_CNN_TRAININGPLOT_WAITDIALOG_TEXT', ...
                'HorizontalAlignment', 'center');
            
            defaultFigurePosition = get(groot, 'DefaultFigurePosition');
            defaultFigurePosition(3:4) = [figureWidth figureHeight];
            h.Position = defaultFigurePosition;
            
            drawnow();
        end
        
        function h = createMessageBox(~, message, titleMessage, tag)
            h = msgbox(message.getString(), titleMessage.getString(), 'warn', 'modal');
            h.Tag = tag;
        end
    end
    
end

