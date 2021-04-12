classdef (Abstract) Constraint < handle & matlab.mixin.Heterogeneous
    % Constraint    Base class to create constraint object to be used by
    %               analyzeNetwork to detect issues.
    %
    % To create a set of constraints, derive a new class from this class.
    % The methods within the derived class whose names start with "test"
    % will be automatically called by this class.
    % 
    % Public methods:
    %   this.applyConstraints(networkAnalyzer)
    %                       Applies the constraints in the derived class
    %                       to the object "networkAnalyzer".
    %                       All the methods whose name starts with "test"
    %                       will be called. The methods take no arguments,
    %                       and should rely on the protected methods of
    %                       the base class to apply its constraints.
    %                       The methods should add errors and warning to
    %                       the layers using the protected methods of the
    %                       base class.
    %
    % Protected methods:
    %   this.addLayerError(index, message)
    %                       Adds the error in the string "message" to the
    %                       layer in the "index" position in the
    %                       LayerAnalyzer array.
    %   this.addLayerWarning(index, message)
    %                       Adds the warning in the string "message" to the
    %                       layer in the "index" position in the
    %                       LayerAnalyzer array.
    % 
    % Protected properties:
    %   NetworkAnalyzer     The NetworkAnalyzer object to which the
    %                       constraint is being applied.
    %   LayerAnalyzers      Convenience property. The following two
    %                       statements are equivalent:
    %                           >> test.NetworkAnalyzer.LayerAnalyzers
    %                           >> test.LayerAnalyzers
    %   Connections         Convenience property. The following two
    %                       statements are equivalent:
    %                           >> test.NetworkAnalyzer.Connections
    %                           >> test.Connections
    %   InternalConnections Convenience property. The following two
    %                       statements are equivalent:
    %                           >> test.NetworkAnalyzer.InternalConnections
    %                           >> test.InternalConnections
    
    %   Copyright 2017 The MathWorks, Inc.
    
    
    
    properties ( GetAccess = protected, SetAccess = private )
        
        NetworkAnalyzer nnet.internal.cnn.analyzer.NetworkAnalyzer = ...
            iEmptyNetworkAnalyzer();
        
        Connections table = iEmptyConnections();
        InternalConnections(:,4) double = iEmptyInternalConnections();
        
        Issues table;
        
    end
    
    properties ( Dependent, GetAccess = protected, SetAccess = private )
        
        LayerAnalyzers(:,1) nnet.internal.cnn.analyzer.util.LayerAnalyzer;
        
    end
    
    methods ( Sealed )
        
        function issues = applyConstraints(this, networkAnalyzer)
            % Applies the constraints in the derived class
            % to the object "networkAnalyzer".
            %
            % All the methods whose name starts with "test"
            % will be called. The methods take no arguments,
            % and should rely on the protected properties of
            % the base class to apply its constraints.
            % The methods should add errors and warning to
            % the layers using the protected methods of the
            % base class.
            
            issues = iEmptyIssuesTable();

            for i=1:numel(this)
                constraint = this(i);
                
                constraint.NetworkAnalyzer = networkAnalyzer;
                constraint.Issues = iEmptyIssuesTable();

                constraint.Connections = networkAnalyzer.Connections;
                constraint.InternalConnections = networkAnalyzer.InternalConnections;

                classinfo = metaclass(constraint);
                methods = string({classinfo.MethodList.Name});
                tests = startsWith(methods, "test");

                testMethods = methods(tests);

                for m=1:numel(testMethods)
                    constraint.(testMethods{m})();
                end

                networkAnalyzer = constraint.NetworkAnalyzer;

                issues = [issues; constraint.Issues]; %#ok<AGROW>

                constraint.NetworkAnalyzer = iEmptyNetworkAnalyzer();
                constraint.Issues = iEmptyIssuesTable();

                constraint.Connections = iEmptyConnections();
                constraint.InternalConnections = iEmptyInternalConnections();
            end
        end
        
    end
    
    methods ( Access = protected )
        
        function addLayerErrorWithId(this, layerIndex, id, msg, varargin)
            if ~isa(msg, 'message')
                [msg, msgId] = iGetMessage(msg, varargin{:});
                if id == ""
                    id = msgId;
                end
            end
            this.addIssueWithId("E", "Layer", layerIndex, id, msg, varargin{:});
        end
        function addLayerWarningWithId(this, layerIndex, id, msg, varargin)
            if ~isa(msg, 'message')
                [msg, msgId] = iGetMessage(msg, varargin{:});
                if id == ""
                    id = msgId;
                end
            end
            this.addIssueWithId("W", "Layer", layerIndex, id, msg, varargin{:});
        end
        
        function addLayerError(this, layerIndex, msg, varargin)
            this.addLayerErrorWithId(layerIndex, "", msg, varargin{:});
        end
        
        function addLayerWarning(this, layerIndex, msg, varargin)
            this.addLayerWarningWithId(layerIndex, "", msg, varargin{:});
        end
        
        function addIssue(this, severity, type, layers, msg, varargin)
            this.addIssueWithId(severity, type, layers, "", msg, varargin{:});
        end

        function addIssueWithId(this, severity, type, layers, id, msg, varargin)
            if islogical(layers) || isnumeric(layers)
                layers = string({this.LayerAnalyzers(layers).Name}');
            end
            names = string({this.LayerAnalyzers(layers).DisplayName}');
            
            userData = struct();
            if ~isempty(varargin) && isstruct(varargin{end})
                userData = varargin{end};
                varargin(end) = [];
            end
            
            try
                [msg, msgId] = iGetMessage(msg, varargin{:});
                if id == ""
                    id = msgId;
                end
            catch
            end
            
            this.Issues = [this.Issues
                {{layers}, {names}, severity, type, id, msg, userData}];
        end
        
    end
    
    methods
        
        function v = get.LayerAnalyzers(this)
            v = this.NetworkAnalyzer.LayerAnalyzers;
        end
        function set.LayerAnalyzers(this, v)
            this.NetworkAnalyzer.LayerAnalyzers = v;
        end
        
    end
    
    methods ( Static )
        
        function constraints = getBuiltInConstraints()
            mc = ?nnet.internal.cnn.analyzer.constraints.Constraint;
            constraints = string({mc.ContainingPackage.ClassList.Name}');
            constraints(constraints == mc.Name) = [];
            constraints = cellfun(@feval, constraints, 'UniformOutput', false);
            constraints = [constraints{:}]';
        end
        
    end
    
end

function [msg, id] = iGetMessage(id, varargin)
    % Check if we received a userData struct, and remove it
    if ~isempty(varargin) && isstruct(varargin{end})
        varargin(end) = [];
    end

    % Check if the message has a list
    list = string.empty();
    try
        for i=1:numel(varargin)
            list(end+1,1) = iGetList(id, i, varargin{end}); %#ok<AGROW>
            varargin(end) = [];
        end
    catch
    end
    
    try
        % Try obtaining a message within the NetworkAnalyzer namespace
        msg = iMessage(id, varargin{:});
        id = msg.Identifier;
        msg = string(msg);
    catch
        % Try obtaining a message in the global namespace
        msg = message(id, varargin{:});
        id = msg.Identifier;
        msg = string(msg);
    end
    
    % If the message has a list, append the list to the message.
    if ~isempty(list)
        msg = msg + newline + strjoin(list, newline);
    end
end

function list = iGetList(id, append, list)
    append = string(append);
    append(append == "1") = "";

    limit = 5;
    
    header = string(iMessage(id + "ListHeader" + append));
    
    nList = size(list, 1);
    n = min(limit, nList);
    
    item = strings(n,1);
    for i = 1:n
        item(i) = string(iMessage(id + "ListItem" + append, list{i,:}));
    end
    
    if n >= limit
        item(limit) = string(iMessage(id + "ListPlus" + append, nList-n+1));
    end
    
    list = header + newline + strjoin("    " + item, newline);
end

function msg = iMessage(id, varargin)
    id = "nnet_cnn:internal:cnn:analyzer:constraints:" + id;
    msg = message(id{1}, varargin{:});
end

function o = iEmptyNetworkAnalyzer()
    o = nnet.internal.cnn.analyzer.NetworkAnalyzer.empty();
end

function o = iEmptyConnections()
    o = table({},{}, 'VariableNames', {'Source', 'Destination'});
end

function o = iEmptyInternalConnections()
    o = double.empty(0,4);
end

function o = iEmptyIssuesTable()
    o = table( {},{},{},{},{},{},{}, 'VariableNames', ...
        {'Layers', 'LayerDisplayNames', 'Severity', 'Type', 'Id', 'Message', 'UserData'} );
end