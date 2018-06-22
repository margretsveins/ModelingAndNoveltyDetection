% function [Etrain_arr, Etest_arr, K_vec, y, sig2] = model2_new(E, E_train,nits,varargin)
function [K_vec, y, sig2] = model2_new(E, E_train,nits,varargin)
% Get varargin
% only want 5 optional inputs at most
numvarargs = length(varargin);
if numvarargs > 7
    error('model2 requires at most 7 optional inputs');
end
% set defaults for optional inputs (no plot and use all the PC)
optargs = {'Plot' false};
optargs(1:numvarargs) = varargin;
[Plot, Plot_boleen] = optargs{:};


% Initialize parameters
D=size(E,1);                    % Dimension of data
Ntrain_initial=size(E_train,2);         % number of training windows
Ntest= size(E,2)-Ntrain_initial;       % number of test examples   
method = 1;


K = 2;
E = E';
% [E_train,E_test]=getdata(1000,500,0.9, 0.9);
Training_set = E_train';
randn('seed',0)

% Define train and test set
E_train = E(1:Ntrain_initial,:);
E_test = E(Ntrain_initial+1:end, :);

random_vector = randsample(Ntrain_initial,Ntrain_initial);
E_train = Training_set(random_vector(1:2),:);

[y,sig2,prob_k]=gm_init(E_train,K,method);

E2train = ones(K,1)*sum((E_train.*E_train)');

% Threshold
e_max = ones(1,3600)*0.2;
e_growth = (1:3600)/3600;
e_t = [e_max; e_growth.*e_max];
e_t_min = min(e_t);

Q_t = 2*log(1./e_t_min);
Mahalanobis_dist_all = 0;
n = 1;
% for n = 3:Ntrain_initial
while n < Ntrain_initial & all(Mahalanobis_dist_all <= Q_t(end))
    K_vec(n) = K;
%     new_point = Training_set(random_vector(n),:);
    new_point = Training_set(randsample(Ntrain_initial,1),:);
    E_train = [E_train;new_point];
    E2train = ones(K,1)*sum((E_train.*E_train)');
    Ntrain = size(E_train,1);        
    for t=1:nits     
       prob_k_x = [];
       dist=sum((y.*y)')'*ones(1,Ntrain) + E2train -2*y*E_train';              % || x_n - mu_k ||^2
       prob_x_k=diag(1./((2*pi*sig2).^(D/2)) )*exp(-0.5*diag(1./sig2)*dist);    % p(x|mu_k,sig2_k)
       prob_x=sum(diag(prob_k)*prob_x_k);                                       % p(x|w)
       for k=1:K
          prob_k_x(k,:)=prob_k(k)*prob_x_k(k,:)./prob_x;                        % p(k|x_n) = gamma_{nk}
       end
       y=diag(1./sum(prob_k_x'))*prob_k_x*E_train ;                              % y(k) = mu_k

       dist=sum((y.*y)')'*ones(1,Ntrain) + E2train -2*y*E_train';   
       sig2=(1/D)*diag(1./sum(prob_k_x'))*(sum((dist.*prob_k_x)')')+ 10e-5;
%        sig_arr(:,t)=sig2;
       prob_k=sum(prob_k_x')/Ntrain;                                            % pi_k
  
    end   %end EM
%     Etrain_arr(n)=gm_cost(E_train,y,sig2,prob_k);
%     Etest_arr(n)=gm_cost(E_test,y,sig2,prob_k);  
    Mahalanobis_dist_all = diag(1./sig2)*dist;
     Mahalanobis_dist_all = min(Mahalanobis_dist_all);  
    Mahalanobis_dist = Mahalanobis_dist_all(:,end);
    for q = 1:K
        Mahalanobis_dist1(q) = (new_point-y(q,:))*(1/sig2(q))*(new_point-y(q,:))';
    end 
    if min(Mahalanobis_dist) >= Q_t(n)
           K = K+1;
           y = [y; new_point];
           [~, l] = max(prob_k_x(:,end));
           y_l = y(l,:);
           C = (y-y_l)*(y-y_l)';
           sig2_new = trace(C)/D;
           sig2 = [sig2; sig2_new];
           p=ones(K,1)/K;
           prob_k = p; 
    end 
    
    n = n + 1
end 
frac_worst = 0.03;
num_window = size(E,1);
[E_sum_train prob_train]=gm_cost(E_train,y,sig2,prob_k);
[E_sum_test prob_test]=gm_cost(E_test,y,sig2,prob_k);

E_prob = [prob_train prob_test];

E_prob_sort = sort(E_prob);

end 