classdef Communicator < handle
% Communicator  Lifetime management of an MPI communicator and allow
% switching

    properties (Access = private)
        Comm
    end
    
    methods
        function obj = Communicator( color )
            obj.Comm = mpiCommManip( 'split', color, labindex );
        end
        
        function cleanupObj = select( this )
            originalComm = mpiCommManip( 'select', this.Comm );
            cleanupObj = onCleanup( @()mpiCommManip( 'select', originalComm ) );
        end
        
        function delete( this )
            mpiCommManip('free', this.Comm);
        end
    end
    
end