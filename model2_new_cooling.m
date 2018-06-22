% function [Etrain_arr, Etest_arr, K_vec, y, sig2] = model2_new(E, E_train,nits,varargin)
function [K_vec, y, sig2,prob_k, Mahalanobis_dist_vec, Q_t_vec, y_cell, sig2_cell, prob_k_cell] = model2_new_cooling(E_train,varargin)
% Get varargin
% only want 5 optional inputs at most
numvarargs = length(varargin);
if numvarargs > 7
    error('model2 requires at most 7 optional inputs');
end
% set defaults for optional inputs (no plot and use all the PC)
optargs = {'e_max', 0.2, 't_alpha', 1, 'aplpha_0', 0.7};
optargs(1:numvarargs) = varargin;
[~, e_max_value, ~, t_alpha, ~, alpha_0] = optargs{:};


% Initialize parameters
D=size(E_train,1);                    % Dimension of data
Ntrain_initial=size(E_train,2);         % number of training points

% Transpose data 
Training_set = E_train';
randn('seed',0)

% Get the first data point 
% x_inital = Training_set(randsample(Ntrain_initial,1),:);
method = 2;
K = 1;
[y,sig2,prob_k]=gm_init(Training_set,K,method);

x_t = y;

% Calculate p(x|k), p(x), p(k|x)
% p(x|k:)
prob_x_k=diag(1./((2*pi*sig2).^(D/2)) ) * exp(-0.5*diag((x_t' - y')'*(x_t' - y'))./sig2);    
% p(x)
prob_x=sum(diag(prob_k)*prob_x_k);                                      
% p(k|x)
for k=1:K
   prob_k_x(k,:)=prob_k(k)*prob_x_k(k,:)./prob_x;                       
end    

% Parameter for while loop
n = 1;
t = 1;
n_iter = 1;
prob_k_x_matrix = prob_k_x;
y_cell{1,1} = y;
sig2_cell{1,1} = sig2;
prob_k_cell{1,1} = prob_k;

% while min(t) << Ntrain
while min(t) < Ntrain_initial*3 
    x_t = Training_set(randsample(Ntrain_initial,1),:);
    Ntrain = size(x_t,1);    
    
% ---------------------- Network growth -------------------------------
    % Get threshold
    e_max = ones(1,size(t,1))*e_max_value;
    e_t = [e_max; t'/Ntrain_initial.*e_max];
    e_t = min(e_t);
    
    Q_t = 2*log(1./e_t);    
    
     % Calculate mahalanobis dist
    Mahalanobis_dist = diag((x_t' - y')'*(x_t' - y'))./sig2;   
    [Mahalanobis_dist mahala_index] = min(Mahalanobis_dist);
    Mahalanobis_dist_vec(n) = Mahalanobis_dist;
    Q_t_vec(n) = Q_t(mahala_index);
    
    if Mahalanobis_dist >= Q_t(mahala_index)
       % Find the new sig2 p.275
       [~, l] = max(prob_k_x(:,end));
       y_l = y(l,:);
       C = (x_t-y_l)*(x_t-y_l)';
       sig2_new = trace(C)/D;
%        if sum(isinf(sig2_new)) > 0 || sum(isnan(sig2_new)) > 0
%              test = 1;
%        end 
       
%        if sig2_new < 3*max(sig2)
           % Update K 
           K = K+1;

           % Add the new center 
           y = [y; x_t];       

           sig2 = [sig2; sig2_new];

           % Update the prior probability so they will be equal
           p=ones(K,1)/K;
           prob_k = p; 

           % Update t
           t = [t;0];
           
           % Get a new random point x from X
%            x_t = Training_set(randsample(Ntrain_initial,1),:);
%        end 
%           prob_x_k=diag(1./((2*pi*sig2).^(D/2)) ) * exp(-0.5*diag((x_t' - y')'*(x_t' - y'))./sig2);
%           prob_k_x_matrix = prob_x_k;
%           n_iter = 0;
    end 
    
    % Keep track of K
    K_vec(n) = K;                
    
    % Calculate p(x|k), p(x), p(k|x)
    % p(x|k:)
    prob_x_k=diag(1./((2*pi*sig2).^(D/2)) ) * exp(-0.5*diag((x_t' - y')'*(x_t' - y'))./sig2);    
    % p(x)
    prob_x=sum(diag(prob_k)*prob_x_k);                                      
    % p(k|x)
    for k=1:K
       prob_k_x(k,:)=prob_k(k)*prob_x_k(k,:)./prob_x;                       
    end
    
    % ------ Update y and sig2 according to (2.9) and (2.10) --------------
    y_t = y;
    sig2_t = sig2;
    
    % Get alpha_t
    a_t = alpha_0./(t+t_alpha);
    
    % y_t+1
    y_t_plus_1 = (y_t + a_t.*(prob_k_x*x_t-y_t))./(ones(K,D).*((1-a_t) + a_t.*prob_k_x)); 
    

    % sig2_t+1
    sig2_t_plus_1 = (sig2_t + a_t.*(diag(prob_k_x)*diag((x_t-y_t_plus_1)*(x_t-y_t_plus_1)')-sig2_t))./((1-a_t) + a_t.*prob_k_x);
    
%     if sum(isinf(sig2_t_plus_1)) > 0 || sum(isnan(sig2_t_plus_1)) > 0
%         test = 1;
%     end 
        
    y = y_t_plus_1;
    sig2  = sig2_t_plus_1;    
    
    % Update p(k)
    prob_k_t = prob_k;
    
    % Use alpha_t according to article
    prob_k_t_plus_1 = prob_k_t + a_t.*( prob_k_x - prob_k_t);
     
    % Normalize
    prob_k_t_plus_1 = prob_k_t_plus_1/sum(prob_k_t_plus_1);
%     
    prob_k = prob_k_t_plus_1;
    
%     %AS THE EM ALGORTIHM
%     prob_k_x_matrix = [prob_k_x_matrix prob_k_x];
%     n_iter = n_iter + 1;
%     
%     prob_k_t_plus_new = (sum(prob_k_x_matrix'))/n_iter;    
%       
%     prob_k = prob_k_t_plus_new';
%     
%     if sum(prob_k_t_plus_new) ~= 1     
%         sum(prob_k_t_plus_new)
%     end
     
    
    y_cell{1,n+1} = y;
    sig2_cell{1,n+1} = sig2;
    prob_k_cell{1,n+1} = prob_k;

    % ----------------- Update loop parameters ----------------------------
    t = t + 1;
    n = n + 1;    
    at_pre = a_t;
end 



end 