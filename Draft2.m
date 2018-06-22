1/(2*pi)^(D/2)*det(diag(sig2(:,:)))^0.5 * exp(-0.5*(x_t' - y(:,:)')*inv(diag(sig2(:,:)))*(x_t' -y(:,:)')')

(x_t - y)'*inv(diag(sig2))*(x_t -y),


(x_t - y(1,:))*(x_t - y(1,:))';


%% 
(x_t - y(1,:))*inv(diag(sig2(1,:)))*(x_t -y(1,:))'
(x_t - y(2,:))*inv(diag(sig2(2,:)))*(x_t -y(2,:))'

%% 
(x_t - y(1,:))
(x_t - y(2,:))

(x_t - y)

%% 
(x_t' - y(1,:)')'*(x_t' - y(1,:)')
(x_t' - y(2,:)')'*(x_t' - y(2,:)')

diag((x_t' - y')'*(x_t' - y'))

%%
(x_t' - y(1,:)')'*(x_t' - y(1,:)')./sig2(1,:)
(x_t' - y(2,:)')'*(x_t' - y(2,:)')./sig2(2,:)

diag((x_t' - y')'*(x_t' - y'))./sig2
%% 
 % Get a random point x from X
    x_t = Training_set(randsample(Ntrain_initial,1),:);
    Ntrain = size(x_t,1); 
    prob_k_t = prob_k;
    % Calculate p(x|k), p(x), p(k|x)
    % p(x|k:)
    prob_x_k_t=diag(1./((2*pi*sig2).^(D/2)) ) * exp(-0.5*diag((x_t' - y')'*(x_t' - y'))./sig2);   
    % p(x)
    prob_x_t=sum(diag(prob_k_t)*prob_x_k_t);                                       % p(x|w)
    % p(k|x)
    for k=1:K
       prob_k_x_t(k,:)=prob_k_t(k)*prob_x_k_t(k,:)./prob_x_t;                        % p(k|x_n) = gamma_{nk}
    end
    
E_train  = repmat(x_t, 30,1);
Ntrain=size(E_train,1);  
E2train = ones(K,1)*sum((E_train.*E_train)');
   dist=sum((y.*y)')'*ones(1,Ntrain) + E2train -2*y*E_train';                % || x_n - mu_k ||^2
   prob_x_k=diag(1./((2*pi*sig2).^(D/2)) )*exp(-0.5*diag(1./sig2)*dist);    % p(x|mu_k,sig2_k)
   prob_x=sum(diag(prob_k)*prob_x_k);                                       % p(x|w)
   for k=1:K
      prob_k_x(k,:)=prob_k(k)*prob_x_k(k,:)./prob_x;                        % p(k|x_n) = gamma_{nk}
   end,
   
   
 %% 

(y_t(1,:) + a_t*(prob_k_x(1,:)*x_t-y_t(1,:)))/((1-a_t) + a_t*prob_k_x(1,:))
(y_t(2,:) + a_t*(prob_k_x(2,:)*x_t-y_t(2,:)))/((1-a_t) + a_t*prob_k_x(2,:)) 
 
(y_t + a_t*(prob_k_x*x_t-y_t))./((1-a_t) + a_t*prob_k_x)  
 
%% 

(sig2_t(1,:) + a_t*(prob_k_x(1,:)*(x_t-y_t(1,:))*(x_t-y_t(1,:))'-sig2_t(1,:)))/((1-a_t) + a_t*prob_k_x(1,:))
(sig2_t(2,:) + a_t*(prob_k_x(2,:)*(x_t-y_t(2,:))*(x_t-y_t(2,:))'-sig2_t(2,:)))/((1-a_t) + a_t*prob_k_x(2,:))

(sig2_t + a_t*(diag(prob_k_x)*diag((x_t-y_t)*(x_t-y_t)')-sig2_t))./((1-a_t) + a_t*prob_k_x)
 