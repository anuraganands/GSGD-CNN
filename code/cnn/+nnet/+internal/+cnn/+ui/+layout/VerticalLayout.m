classdef VerticalLayout < nnet.internal.cnn.ui.layout.Layout
    % VerticalLayout   Container that lays out its children vertically with heights determined by their weights
    %
    % If any of the children become invisible, the children are laid out
    % again so that the invisible child no longer takes up space.
    
    %   Copyright 2017 The MathWorks, Inc.
    
    properties(SetAccess = private)
        % Parent   (graphics handle) The parent component
        Parent
        
        % MainPanel   (matlab.ui.container.Panel)
        % The main panel where all the laid out children will reside
        MainPanel
        
        % Weights   (double array) The weights which determine the relative
        % heights of each child. If a child becomes invisible, then the
        % remaining visible children are laid out so that their heights
        % maintain their previous ratios.
        Weights
        
        % Children   (array of graphics handles) The children in the order
        % they were added. Note that this is not the same as the
        % MainPanel.Children because MainPanel.Children has a different
        % ordering and may also have a smaller number of objects (if they
        % are made invisible).
        Children = []
    end
    
    properties(Access = private)
        % VisibilityListeners    (cell of listeners) Listeners on the
        % Children for changes to Visible property
        VisibilityListeners = {}
        
        % ReparentingListeners   (cell of listeners) Listeners on the
        % Children for changes to Parent property
        ReparentingListeners = {}
    end
    
    methods
        function this = VerticalLayout(parent)
            this.Parent = parent; 
            this.MainPanel = uipanel('Parent', parent, 'BorderType', 'none', 'Tag', 'NNET_CNN_TRAININGPLOT_VERTICALLAYOUT_MAINPANEL');
        end
        
        function add(this, child, weight)
            child.Parent = this.MainPanel;
            this.Weights = [this.Weights; weight];
            this.Children = [this.Children; child];
            
            child.Units = 'normalized';
            this.updatePositions();
            this.VisibilityListeners{end+1} = addlistener(child, 'Visible', 'PostSet', @this.visibleChangedCallback);
            this.ReparentingListeners{end+1} = addlistener(child, 'Parent', 'PostSet', @this.parentChangedCallback);
        end
    end
    
    methods(Access = private)
        function updatePositions(this)
            heightsOfChildren = this.computeNormalizedVisibleWeights();
            heightsFromBottom = iComputeSumOfHeightsOfChildrenBeneathEachChild(heightsOfChildren);
            for i=1:numel(this.Children)
                this.Children(i).Position = [0, heightsFromBottom(i), 1, heightsOfChildren(i)]; 
            end
        end
        
        function normalizedVisibleWeights = computeNormalizedVisibleWeights(this)
            % computeNormalizedVisibleWeights   Weights this.Weights by
            % the corresponding children visibilities, then normalizes.
            areChildrenVisible = arrayfun(@(x) iIsVisible(x), this.Children, 'UniformOutput', true); 
            visibleWeights = this.Weights .* double(areChildrenVisible);
            sumOfWeights = sum(visibleWeights);
            if sumOfWeights == 0
                normalizedVisibleWeights = zeros(numel(this.Weights), 1); 
            else
                normalizedVisibleWeights = visibleWeights / sum(visibleWeights);
            end
        end
        
        % callbacks
        function visibleChangedCallback(this, ~, ~)
            this.updatePositions();
        end
        
        function parentChangedCallback(this, ~, event)
            childIndex = find(this.Children == event.AffectedObject, 1);
            assert(~isempty(childIndex), 'Reparented child should actually be a member of this.Children!');
            
            this.Weights(childIndex) = [];
            this.Children(childIndex) = [];
            
            delete(this.VisibilityListeners{childIndex});
            this.VisibilityListeners(childIndex) = [];
            
            delete(this.ReparentingListeners{childIndex});
            this.ReparentingListeners(childIndex) = [];
            
            this.updatePositions();
        end
    end
end

% helpers
function tf = iIsVisible(x)
tf = strcmp(x.Visible, 'on');
end

function heights = iComputeSumOfHeightsOfChildrenBeneathEachChild(heightsOfChildren)
% For each child, compute the sum of heights of all children beneath it.
% For example, if the heights are [h1, h2, h3, h4], then we should end up
% with [h2+h3+h4, h3+h4, h4, 0].
heightsOfChildren(1) = [];
heights = cumsum([heightsOfChildren; 0], 'reverse');
end
