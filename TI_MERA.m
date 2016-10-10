function TI_MERA
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Translationally Invariant MERA
%
% Andrew Goldsborough - 21/09/2016
%
% basic MERA algorithm for the 1D Quantum Ising model
% for details see arXiv:0707.1454v4
%
% index convention
%    -1      -1  -3
%     |       |___|
%   /_ _\     |___|
%   | | |     |   |
% -2 -3 -4   -2  -4
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%inputs
num_levels = 5;     %number of levels 
L = 2*3^num_levels; %chain length
lambda = 1.0;       %magnetic field
chi = 8;            %max chi
ULmax = 1;          %number of updates on single tensor
optmax = 1;         %number of updates on tensor layer
sweepmax = 50;      %number of full network sweeps

%turn off warnings
warning('off','MATLAB:eigs:SigmaChangedToSA');
warning('off','ncon:suboptimalsequence');

%import Quantum Ising model
[h,d,shift] = QIsing_twosite(lambda);
h0 = h;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%initialise tensors
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%storage for tensors
w = cell(num_levels,1);
u = cell(num_levels,1);
rho = cell(num_levels,1);

%generate random start tensors
level = 1;
u{level} = u_normalise(rand(d,d,d,d));
w{level} = w_normalise(rand(min(chi,d^3),d,d,d));

for level = 2:num_levels
    u{level} = u_normalise(rand(size(w{level-1},1),size(w{level-1},1),size(w{level-1},1),size(w{level-1},1)));
    w{level} = w_normalise(rand(min(chi,size(w{level-1},1)^3),size(w{level-1},1),size(w{level-1},1),size(w{level-1},1)));
end

top = top_normalise(rand(size(w{num_levels},1),size(w{num_levels},1)));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%sweep over the blocks
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for sweep = 1:sweepmax
    %update rho
    rho{num_levels,1} = ncon({conj(top),top},{[-1,-3],[-2,-4]});
    
    for level = num_levels-1:-1:1
        rho{level} = descend(rho{level+1},u{level+1},w{level+1});
    end
        
    %reset h
    h = h0;
    
    %update tensors
    for level = 1:num_levels
        
        %update tensors
        [u{level},w{level}] = MERAupdate(u{level},w{level},h,rho{level},optmax,ULmax);
        
        %raise Hamiltonian for next level
        h = ascend(h,u{level},w{level});
    end
    
    %top tensor
    h = tfuse(tfuse(h+permute(h,[3,4,1,2]),[1,-2,1,-3]),[-1,2,2]);
    [V,hspec] = eig(0.5*(h+h'));
    [~,idx] = sort(diag(hspec));
    top = tsplit(V(:,idx(1)),1,[size(w{num_levels},1),size(w{num_levels},1)]);
    fprintf('Sweep: %d, Energy: %.15e\n',sweep,hspec(idx(1),idx(1))/L + shift);
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [h,d,shift] = QIsing_twosite(lambda)
%creates the two site quantum Ising model Hamiltonian
%note energy shift for use with MERA

pauli_x = [0 1; 1 0];
pauli_z = [1 0; 0 -1];

%h = XX + 0.5*lambda (ZI + IZ)
h = ncon({pauli_x,pauli_x},{[-1,-2],[-3,-4]})...
    + 0.5*lambda*(ncon({pauli_z,eye(2)},{[-1,-2],[-3,-4]})...
    + ncon({eye(2),pauli_z},{[-1,-2],[-3,-4]}));

%shift energy spectrum by largest eigenvalue
shift = max(eig(tfuse(tfuse(h,[1,-2,1,-3]),[-1,2,2])));
h = h - shift * ncon({eye(2),eye(2)},{[-1,-2],[-3,-4]});

%dimension of spin = 2
d = 2;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function u = u_normalise(u)
%normalises unitary

[uleg1,uleg2,uleg3,uleg4] = size(u);
[U,~,V] = svd(tfuse(tfuse(u,[-2,1,-3,1]),[-1,2,2]),'econ');
u = -V*U';

%split into a 4-index tensor
u = tsplit(u,1,[uleg1,uleg3]);
u = tsplit(u,3,[uleg2,uleg4]);
u = permute(u,[1,3,2,4]);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function w = w_normalise(w)
%normalises isometry

[~,wleg2,wleg3,wleg4] = size(w);
[U,~,V] = svd(tfuse(w,[-2,1,1,1]),'econ');
w = -V*U';

%split into a 4-index tensor
w = tsplit(w,2,[wleg2,wleg3,wleg4]);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function top = top_normalise(top)
%normalises top tensor
norm = ncon({top,top},{[1,2],[1,2]});
top = top./sqrt(norm);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function rho = descend(rho,u,w)
%descending superoperator, used to lower projectors

%diagram 1 - left
rho_1 = ncon({rho,w,w,u,conj(u),conj(w),conj(w)},...
    {[5,10,4,3],[10,11,-2,12],[3,8,1,2],[12,-4,8,9],[6,-3,7,9],[5,11,-1,6],[4,7,1,2]});

%diagram 2 - centre
rho_2 = ncon({rho,w,w,u,conj(u),conj(w),conj(w)},...
    {[8,7,6,5],[7,1,2,11],[5,12,3,4],[11,-2,12,-4],[9,-1,10,-3],[8,1,2,9],[6,10,3,4]});
    
%diagram 3 - right
rho_3 = ncon({rho,w,w,u,conj(u),conj(w),conj(w)},...
    {[4,3,5,10],[3,1,2,8],[10,11,-4,12],[8,9,11,-2],[6,9,7,-1],[4,1,2,6],[5,7,-3,12]});

%use the average when translationally invariant
rho = (1/3) * (rho_1 + rho_2 + rho_3);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function h = ascend(h,u,w)
%ascending superoperator, used to raise Hamiltonian

%diagram 1 - left
h_1 = ncon({w,w,u,h,conj(u),conj(w),conj(w)},...
    {[-1,8,9,10],[-3,11,1,2],[10,6,11,7],[9,4,6,5],[3,5,12,7],[-2,8,4,3],[-4,12,1,2]});

%diagram 2 - centre
h_2 = ncon({w,w,u,h,conj(u),conj(w),conj(w)},...
    {[-1,1,2,11],[-3,9,3,4],[11,7,9,8],[7,5,8,6],[12,5,10,6],[-2,1,2,12],[-4,10,3,4]});

%diagram 3 - right
h_3 = ncon({w,w,u,h,conj(u),conj(w),conj(w)},...
    {[-1,1,2,11],[-3,8,9,10],[11,6,8,7],[7,4,9,5],[12,6,3,4],[-2,1,2,12],[-4,3,5,10]});

h = h_1 + h_2 + h_3;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [u,w] = MERAupdate(u,w,h,rho,optmax,ULmax)
%updates u and w for a layer in a TI MERA

%useful values
[uleg1,uleg2,uleg3,uleg4] = size(u);
[~,wleg2,wleg3,wleg4] = size(w);

for optcount = 1:optmax
    %u
    for ULcount = 1:ULmax
        
        %diagram 1 - left
        envu_1 = ncon({rho,w,w,h,conj(u),conj(w),conj(w)},...
            {[5,10,4,3],[10,11,12,-2],[3,-4,1,2],[12,8,-1,9],[6,9,7,-3],[5,11,8,6],[4,7,1,2]});
        
        %diagram 2 - centre
        envu_2 = ncon({rho,w,w,h,conj(u),conj(w),conj(w)},...
            {[8,7,6,5],[7,1,2,-2],[5,-4,3,4],[-1,11,-3,12],[9,11,10,12],[8,1,2,9],[6,10,3,4]});
        
        %diagram 3 - right
        envu_3 = ncon({rho,w,w,h,conj(u),conj(w),conj(w)},...
            {[4,3,5,10],[3,1,2,-2],[10,-4,11,12],[-3,8,11,9],[6,-1,7,8],[4,1,2,6],[5,7,9,12]});
        
        %optimise u by UL
        envu = envu_1 + envu_2 + envu_3;
        envu = tfuse(envu,[1,-2,1,-3]);
        envu = tfuse(envu,[-1,2,2]);
        
        [U,~,V] = svd(envu,'econ');
        u = -V*U';
        
        %split into a 4-index tensor
        u = tsplit(u,1,[uleg1,uleg3]);
        u = tsplit(u,3,[uleg2,uleg4]);
        u = permute(u,[1,3,2,4]);        
    end
    
    %w
    for ULcount = 1:ULmax
        
        %diagram 1 - w right, h left
        envw_1 = ncon({rho,w,u,h,conj(u),conj(w),conj(w)},...
            {[10,9,12,-4],[9,6,7,8],[8,4,-1,5],[7,2,4,3],[1,3,11,5],[10,6,2,1],[12,11,-2,-3]});
        
        %diagram 2 - w right, h centre
        envw_2 = ncon({rho,w,u,h,conj(u),conj(w),conj(w)},...
            {[10,9,12,-4],[9,1,2,7],[7,5,-1,6],[5,3,6,4],[8,3,11,4],[10,1,2,8],[12,11,-2,-3]});
        
        %diagram 3 - w right, h right
        envw_3 = ncon({rho,w,u,h,conj(u),conj(w),conj(w)},...
            {[4,3,5,-4],[3,1,2,10],[10,11,-1,12],[12,8,-2,9],[6,11,7,8],[4,1,2,6],[5,7,9,-3]});
    
        %diagram 4 - w left, h left
        envw_4 = ncon({rho,w,u,h,conj(u),conj(w),conj(w)},...
            {[5,-4,4,3],[3,10,1,2],[-3,11,10,12],[-2,8,11,9],[6,9,7,12],[5,-1,8,6],[4,7,1,2]});
        
        %diagram 5 - w left, h centre
        envw_5 = ncon({rho,w,u,h,conj(u),conj(w),conj(w)},...
            {[12,-4,10,9],[9,7,1,2],[-3,5,7,6],[5,3,6,4],[11,3,8,4],[12,-1,-2,11],[10,8,1,2]});
        
        %diagram 6 - w left, h right
        envw_6 = ncon({rho,w,u,h,conj(u),conj(w),conj(w)},...
            {[12,-4,10,9],[9,6,7,8],[-3,4,6,5],[5,2,7,3],[11,4,1,2],[12,-1,-2,11],[10,1,3,8]});
        
        %optimise w by UL
        envw = envw_1 + envw_2 + envw_3 + envw_4 + envw_5 + envw_6;
        envw = tfuse(envw,[1,1,1,-2]);
        
        [U,~,V] = svd(envw,'econ');
        w = -V*U';
        
        %split into a 4-index tensor
        w = tsplit(w,2,[wleg2,wleg3,wleg4]);
    end
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%optimal contraction orders using netcon
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% rho_1_netcon(netcon({[11,1,12,2],[1,3,-2,4],[2,5,6,7],[4,-4,5,8],[9,-3,10,8],[11,3,-1,9],[12,10,6,7]},0,2,1,1)) = 1:12;
% rho_1_order = {[rho_1_netcon(11),rho_1_netcon(1),rho_1_netcon(12),rho_1_netcon(2)],...
%     [rho_1_netcon(1),rho_1_netcon(3),-2,rho_1_netcon(4)],...
%     [rho_1_netcon(2),rho_1_netcon(5),rho_1_netcon(6),rho_1_netcon(7)],...
%     [rho_1_netcon(4),-4,rho_1_netcon(5),rho_1_netcon(8)],...
%     [rho_1_netcon(9),-3,rho_1_netcon(10),rho_1_netcon(8)],...
%     [rho_1_netcon(11),rho_1_netcon(3),-1,rho_1_netcon(9)],...
%     [rho_1_netcon(12),rho_1_netcon(10),rho_1_netcon(6),rho_1_netcon(7)]};
% 
% rho_2_netcon(netcon({[11,1,12,2],[1,3,4,5],[2,6,7,8],[5,-2,6,-4],[9,-1,10,-3],[11,3,4,9],[12,10,7,8]},0,2,1,1)) = 1:12;
% rho_2_order = {[rho_2_netcon(11),rho_2_netcon(1),rho_2_netcon(12),rho_2_netcon(2)],...
%     [rho_2_netcon(1),rho_2_netcon(3),rho_2_netcon(4),rho_2_netcon(5)],...
%     [rho_2_netcon(2),rho_2_netcon(6),rho_2_netcon(7),rho_2_netcon(8)],...
%     [rho_2_netcon(5),-2,rho_2_netcon(6),-4],...
%     [rho_2_netcon(9),-1,rho_2_netcon(10),-3],...
%     [rho_2_netcon(11),rho_2_netcon(3),rho_2_netcon(4),rho_2_netcon(9)],...
%     [rho_2_netcon(12),rho_2_netcon(10),rho_2_netcon(7),rho_2_netcon(8)]};
% 
% rho_3_netcon(netcon({[11,1,12,2],[1,3,4,5],[2,6,-4,7],[5,8,6,-2],[9,8,10,-1],[11,3,4,9],[12,10,-3,7]},0,2,1,1)) = 1:12;
% rho_3_order = {[rho_3_netcon(11),rho_3_netcon(1),rho_3_netcon(12),rho_3_netcon(2)],...
%     [rho_3_netcon(1),rho_3_netcon(3),rho_3_netcon(4),rho_3_netcon(5)],...
%     [rho_3_netcon(2),rho_3_netcon(6),-4,rho_3_netcon(7)],...
%     [rho_3_netcon(5),rho_3_netcon(8),rho_3_netcon(6),-2],...
%     [rho_3_netcon(9),rho_3_netcon(8),rho_3_netcon(10),-1],...
%     [rho_3_netcon(11),rho_3_netcon(3),rho_3_netcon(4),rho_3_netcon(9)],...
%     [rho_3_netcon(12),rho_3_netcon(10),-3,rho_3_netcon(7)]};
% 
% h_1_netcon(netcon({[-1,1,2,3],[-3,4,5,6],[3,7,4,8],[2,9,7,10],[11,10,12,8],[-2,1,9,11],[-4,12,5,6]},0,2,1,1)) = 1:12;
% h_1_order = {[-1,h_1_netcon(1),h_1_netcon(2),h_1_netcon(3)],...
%     [-3,h_1_netcon(4),h_1_netcon(5),h_1_netcon(6)],...
%     [h_1_netcon(3),h_1_netcon(7),h_1_netcon(4),h_1_netcon(8)],...
%     [h_1_netcon(2),h_1_netcon(9),h_1_netcon(7),h_1_netcon(10)],...
%     [h_1_netcon(11),h_1_netcon(10),h_1_netcon(12),h_1_netcon(8)],...
%     [-2,h_1_netcon(1),h_1_netcon(9),h_1_netcon(11)],...
%     [-4,h_1_netcon(12),h_1_netcon(5),h_1_netcon(6)]};
% 
% h_2_netcon(netcon({[-1,1,2,3],[-3,4,5,6],[3,7,4,8],[7,9,8,10],[11,9,12,10],[-2,1,2,11],[-4,12,5,6]},0,2,1,1)) = 1:12;
% h_2_order = {[-1,h_2_netcon(1),h_2_netcon(2),h_2_netcon(3)],...
%     [-3,h_2_netcon(4),h_2_netcon(5),h_2_netcon(6)],...
%     [h_2_netcon(3),h_2_netcon(7),h_2_netcon(4),h_2_netcon(8)],...
%     [h_2_netcon(7),h_2_netcon(9),h_2_netcon(8),h_2_netcon(10)],...
%     [h_2_netcon(11),h_2_netcon(9),h_2_netcon(12),h_2_netcon(10)],...
%     [-2,h_2_netcon(1),h_2_netcon(2),h_2_netcon(11)],...
%     [-4,h_2_netcon(12),h_2_netcon(5),h_2_netcon(6)]};
% 
% h_3_netcon(netcon({[-1,1,2,3],[-3,4,5,6],[3,7,4,8],[8,9,5,10],[11,7,12,9],[-2,1,2,11],[-4,12,10,6]},0,2,1,1)) = 1:12;
% h_3_order = {[-1,h_3_netcon(1),h_3_netcon(2),h_3_netcon(3)],...
%     [-3,h_3_netcon(4),h_3_netcon(5),h_3_netcon(6)],...
%     [h_3_netcon(3),h_3_netcon(7),h_3_netcon(4),h_3_netcon(8)],...
%     [h_3_netcon(8),h_3_netcon(9),h_3_netcon(5),h_3_netcon(10)],...
%     [h_3_netcon(11),h_3_netcon(7),h_3_netcon(12),h_3_netcon(9)],...
%     [-2,h_3_netcon(1),h_3_netcon(2),h_3_netcon(11)],...
%     [-4,h_3_netcon(12),h_3_netcon(10),h_3_netcon(6)]};
% 
% envu_1_netcon(netcon({[11,1,12,2],[1,3,4,-2],[2,-4,5,6],[4,7,-1,8],[9,8,10,-3],[11,3,7,9],[12,10,5,6]},0,2,1,1)) = 1:12;
% envu_1_order = {[envu_1_netcon(11),envu_1_netcon(1),envu_1_netcon(12),envu_1_netcon(2)],...
%     [envu_1_netcon(1),envu_1_netcon(3),envu_1_netcon(4),-2],...
%     [envu_1_netcon(2),-4,envu_1_netcon(5),envu_1_netcon(6)],...
%     [envu_1_netcon(4),envu_1_netcon(7),-1,envu_1_netcon(8)],...
%     [envu_1_netcon(9),envu_1_netcon(8),envu_1_netcon(10),-3],...
%     [envu_1_netcon(11),envu_1_netcon(3),envu_1_netcon(7),envu_1_netcon(9)],...
%     [envu_1_netcon(12),envu_1_netcon(10),envu_1_netcon(5),envu_1_netcon(6)]};
% 
% envu_2_netcon(netcon({[11,1,12,2],[1,3,4,-2],[2,-4,5,6],[-1,7,-3,8],[9,7,10,8],[11,3,4,9],[12,10,5,6]},0,2,1,1)) = 1:12;
% envu_2_order = {[envu_2_netcon(11),envu_2_netcon(1),envu_2_netcon(12),envu_2_netcon(2)],...
%     [envu_2_netcon(1),envu_2_netcon(3),envu_2_netcon(4),-2],...
%     [envu_2_netcon(2),-4,envu_2_netcon(5),envu_2_netcon(6)],...
%     [-1,envu_2_netcon(7),-3,envu_2_netcon(8)],...
%     [envu_2_netcon(9),envu_2_netcon(7),envu_2_netcon(10),envu_2_netcon(8)],...
%     [envu_2_netcon(11),envu_2_netcon(3),envu_2_netcon(4),envu_2_netcon(9)],...
%     [envu_2_netcon(12),envu_2_netcon(10),envu_2_netcon(5),envu_2_netcon(6)]};
% 
% envu_3_netcon(netcon({[11,1,12,2],[1,3,4,-2],[2,-4,5,6],[-3,7,5,8],[9,-1,10,7],[11,3,4,9],[12,10,8,6]},0,2,1,1)) = 1:12;
% envu_3_order = {[envu_3_netcon(11),envu_3_netcon(1),envu_3_netcon(12),envu_3_netcon(2)],...
%     [envu_3_netcon(1),envu_3_netcon(3),envu_3_netcon(4),-2],...
%     [envu_3_netcon(2),-4,envu_3_netcon(5),envu_3_netcon(6)],...
%     [-3,envu_3_netcon(7),envu_3_netcon(5),envu_3_netcon(8)],...
%     [envu_3_netcon(9),-1,envu_3_netcon(10),envu_3_netcon(7)],...
%     [envu_3_netcon(11),envu_3_netcon(3),envu_3_netcon(4),envu_3_netcon(9)],...
%     [envu_3_netcon(12),envu_3_netcon(10),envu_3_netcon(8),envu_3_netcon(6)]};
% 
% envw_1_netcon(netcon({[11,1,12,-4],[1,2,3,4],[4,5,-1,6],[3,7,5,8],[9,8,10,6],[11,2,7,9],[12,10,-2,-3]},0,2,1,1)) = 1:12;
% envw_1_order = {[envw_1_netcon(11),envw_1_netcon(1),envw_1_netcon(12),-4],...
%     [envw_1_netcon(1),envw_1_netcon(2),envw_1_netcon(3),envw_1_netcon(4)],...
%     [envw_1_netcon(4),envw_1_netcon(5),-1,envw_1_netcon(6)],...
%     [envw_1_netcon(3),envw_1_netcon(7),envw_1_netcon(5),envw_1_netcon(8)],...
%     [envw_1_netcon(9),envw_1_netcon(8),envw_1_netcon(10),envw_1_netcon(6)],...
%     [envw_1_netcon(11),envw_1_netcon(2),envw_1_netcon(7),envw_1_netcon(9)],...
%     [envw_1_netcon(12),envw_1_netcon(10),-2,-3]};
% 
% envw_2_netcon(netcon({[11,1,12,-4],[1,2,3,4],[4,5,-1,6],[5,7,6,8],[9,7,10,8],[11,2,3,9],[12,10,-2,-3]},0,2,1,1)) = 1:12;
% envw_2_order = {[envw_2_netcon(11),envw_2_netcon(1),envw_2_netcon(12),-4],...
%     [envw_2_netcon(1),envw_2_netcon(2),envw_2_netcon(3),envw_2_netcon(4)],...
%     [envw_2_netcon(4),envw_2_netcon(5),-1,envw_2_netcon(6)],...
%     [envw_2_netcon(5),envw_2_netcon(7),envw_2_netcon(6),envw_2_netcon(8)],...
%     [envw_2_netcon(9),envw_2_netcon(7),envw_2_netcon(10),envw_2_netcon(8)],...
%     [envw_2_netcon(11),envw_2_netcon(2),envw_2_netcon(3),envw_2_netcon(9)],...
%     [envw_2_netcon(12),envw_2_netcon(10),-2,-3]};
% 
% envw_3_netcon(netcon({[11,1,12,-4],[1,2,3,4],[4,5,-1,6],[6,7,-2,8],[9,5,10,7],[11,2,3,9],[12,10,8,-3]},0,2,1,1)) = 1:12;
% envw_3_order = {[envw_3_netcon(11),envw_3_netcon(1),envw_3_netcon(12),-4],...
%     [envw_3_netcon(1),envw_3_netcon(2),envw_3_netcon(3),envw_3_netcon(4)],...
%     [envw_3_netcon(4),envw_3_netcon(5),-1,envw_3_netcon(6)],...
%     [envw_3_netcon(6),envw_3_netcon(7),-2,envw_3_netcon(8)],...
%     [envw_3_netcon(9),envw_3_netcon(5),envw_3_netcon(10),envw_3_netcon(7)],...
%     [envw_3_netcon(11),envw_3_netcon(2),envw_3_netcon(3),envw_3_netcon(9)],...
%     [envw_3_netcon(12),envw_3_netcon(10),envw_3_netcon(8),-3]};
% 
% envw_4_netcon(netcon({[11,-4,12,1],[1,2,3,4],[-3,5,2,6],[-2,7,5,8],[9,8,10,6],[11,-1,7,9],[12,10,3,4]},0,2,1,1)) = 1:12;
% envw_4_order = {[envw_4_netcon(11),-4,envw_4_netcon(12),envw_4_netcon(1)],...
%     [envw_4_netcon(1),envw_4_netcon(2),envw_4_netcon(3),envw_4_netcon(4)],...
%     [-3,envw_4_netcon(5),envw_4_netcon(2),envw_4_netcon(6)],...
%     [-2,envw_4_netcon(7),envw_4_netcon(5),envw_4_netcon(8)],...
%     [envw_4_netcon(9),envw_4_netcon(8),envw_4_netcon(10),envw_4_netcon(6)],...
%     [envw_4_netcon(11),-1,envw_4_netcon(7),envw_4_netcon(9)],...
%     [envw_4_netcon(12),envw_4_netcon(10),envw_4_netcon(3),envw_4_netcon(4)]};
% 
% envw_5_netcon(netcon({[11,-4,12,1],[1,2,3,4],[-3,5,2,6],[5,7,6,8],[9,7,10,8],[11,-1,-2,9],[12,10,3,4]},0,2,1,1)) = 1:12;
% envw_5_order = {[envw_5_netcon(11),-4,envw_5_netcon(12),envw_5_netcon(1)],...
%     [envw_5_netcon(1),envw_5_netcon(2),envw_5_netcon(3),envw_5_netcon(4)],...
%     [-3,envw_5_netcon(5),envw_5_netcon(2),envw_5_netcon(6)],...
%     [envw_5_netcon(5),envw_5_netcon(7),envw_5_netcon(6),envw_5_netcon(8)],...
%     [envw_5_netcon(9),envw_5_netcon(7),envw_5_netcon(10),envw_5_netcon(8)],...
%     [envw_5_netcon(11),-1,-2,envw_5_netcon(9)],...
%     [envw_5_netcon(12),envw_5_netcon(10),envw_5_netcon(3),envw_5_netcon(4)]};
% 
% envw_6_netcon(netcon({[11,-4,12,1],[1,2,3,4],[-3,5,2,6],[6,7,3,8],[9,5,10,7],[11,-1,-2,9],[12,10,8,4]},0,2,1,1)) = 1:12;
% envw_6_order = {[envw_6_netcon(11),-4,envw_6_netcon(12),envw_6_netcon(1)],...
%     [envw_6_netcon(1),envw_6_netcon(2),envw_6_netcon(3),envw_6_netcon(4)],...
%     [-3,envw_6_netcon(5),envw_6_netcon(2),envw_6_netcon(6)],...
%     [envw_6_netcon(6),envw_6_netcon(7),envw_6_netcon(3),envw_6_netcon(8)],...
%     [envw_6_netcon(9),envw_6_netcon(5),envw_6_netcon(10),envw_6_netcon(7)],...
%     [envw_6_netcon(11),-1,-2,envw_6_netcon(9)],...
%     [envw_6_netcon(12),envw_6_netcon(10),envw_6_netcon(8),envw_6_netcon(4)]};
%