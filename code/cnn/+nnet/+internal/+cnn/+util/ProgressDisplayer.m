classdef ProgressDisplayer < nnet.internal.cnn.util.Reporter
    % ProgressDisplayer   Class to display training progress
    
    %   Copyright 2015-2017 The MathWorks, Inc.
    
    properties
        % Frequency   Frequency to report messages expressed in iterations.
        % This can be a scalar or a vector of frequencies
        Frequency = 50
    end
    
    properties(Access = private)
        % Iteration message to be reported
        IterationMessage = ''
        
        % Last message
        LastDisplayedMessage = ''
        
        % Columns   (nnet.internal.cnn.util.ColumnStrategy) Column strategy
        Columns
    end
    
    methods
        function this = ProgressDisplayer( columnStrategy )
            this.Columns = columnStrategy;
        end
        
        function setup( ~ )
        end
        
        function start( this )
            iPrintString(this.Columns.HorizontalBorder);
            iPrintString(this.Columns.Headings);
            iPrintString(this.Columns.HorizontalBorder);
        end
        
        function reportIteration( this, summary )
            msg = this.buildMsgFromSummary( summary );
            % It is necessary to convert the message to char array to be
            % displayed line by line
            msg = msg.char;
            this.storeIterationMessage( msg );
            if iCanPrint(summary.Iteration, this.Frequency)
                this.displayIterationMessage();
            end
        end
        
        function reportEpoch( ~, ~, ~, ~ )
        end
        
        function finish( this, ~ )
            this.displayIterationMessage();
            iPrintString(this.Columns.HorizontalBorder);
        end
    end
    
    methods(Access = private)
        function storeIterationMessage( this, msg )
            this.IterationMessage = msg;
        end
        
        function displayIterationMessage( this )
            % displayIterationMessage   Displays iteration message if it
            % has changed.
            msg = this.IterationMessage;
            if ~strcmp(msg, this.LastDisplayedMessage)
                disp( msg );
            end
            this.LastDisplayedMessage = msg;
        end
        
        function msg = buildMsgFromSummary( this, summary )
            textPieces = iGetTextPiecesFromSummary( summary, this.Columns.Column );
            msg = strjoin( textPieces, iCentralDelimiter);
            msg = iLeftDelimiter + msg + iRightDelimiter;
        end
    end
end

function iPrintString( str )
fprintf( '%s\n', str );
end

function tf = iCanPrint(iteration, frequency)
tf = any(mod(iteration, frequency) == 0) || (iteration == 1) ;
end

function delimiter = iCentralDelimiter()
delimiter = " | ";
end

function delimiter = iLeftDelimiter()
delimiter = "| ";
end

function delimiter = iRightDelimiter()
delimiter = " |";
end

function [summaryValue, format] = iAdjustFormat(name, summaryValue, ...
    format, width)
switch name
    case 'Time'
        summaryValue = iConvertTimeFormat(summaryValue);
    case 'LearnRate'
        if summaryValue < 1e-4
            format = format.replace("f","e");
        end
    case {'Loss', 'RMSE', 'ValidationLoss', 'ValidationRMSE'}
        % bigger than this is difficult to read  
        largeNumber = 1e+6;       
        % precision is the number after . and before conversion character
        precision = regexp(format,'(?<=\.)(.*?)(?=[A-za-z])','match');
        precision = str2num(precision);
        % using min makes it robust to changes of width
        if summaryValue >= min(10^(width+1), largeNumber)
            format = format.replace("f","e");
        % this avoids returning zero for small values
        elseif summaryValue < 10^(-precision)
            format = format.replace("f","e");            
        end     
end
end

function t = iConvertTimeFormat(t)
% iConvertTimeFormat   Convert the time 't' expressed in seconds to
% 'hh:mm:ss' duration
t = seconds( t );
t.Format = 'hh:mm:ss';
end

function textPieces = iGetTextPiecesFromSummary( summary, columns )
textPieces = arrayfun(@(c)iGetColumnText( summary, c ), columns );
end

function columnText = iGetColumnText( summary, column )
% iGetColumnText   Get formatted text for column getting values from
% summary. The returned text will be a string
name = column.Name;
summaryValue = summary.(name);
format = column.Format;
width = column.Width;
[summaryValue, format] = iAdjustFormat(name, summaryValue, format, width);
if isempty(summaryValue)
    columnText = iEmptyTextCell(width);
else
    columnText = sprintf(format, summaryValue);
end
columnText = string( columnText );
end

function textCell = iEmptyTextCell(formatWidth)
textCell = sprintf("%"+formatWidth+"s",'');
end