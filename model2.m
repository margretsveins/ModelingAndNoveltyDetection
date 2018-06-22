function [Etrain_arr, Etest_arr] = model2(E, E_train, K, nits, method, varargin)

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
Ntrain=size(E_train,2);         % number of training windows
Ntest= size(E,2)-Ntrain;       % number of test examples    

E = E';
E_train = E_train';
randn('seed',0)

% Define train and test set
E_train = E(1:Ntrain,:);
E_test = E(Ntrain+1:end, :);

[y,sig2,prob_k]=gm_init(E_train,K,method);

E2train = ones(K,1)*sum((E_train.*E_train)');

% plot data points
if Plot_boleen
    figure()
    h_train = plot(E_train(:,1),E_train(:,2),'.');
    hold on
    h_test = plot(E_test(:,1),E_test(:,2),'m.');
    for k=1:K,
          h_init = plot(y(k,1),y(k,2),'g*'); 
          text(y(k,1),y(k,2),[int2str(k),'-',int2str(0)])
          drawnow
    end
end
for t=1:nits,
   dist=sum((y.*y)')'*ones(1,Ntrain) + E2train -2*y*E_train';                % || x_n - mu_k ||^2
   prob_x_k=diag(1./((2*pi*sig2).^(D/2)) )*exp(-0.5*diag(1./sig2)*dist);    % p(x|mu_k,sig2_k)
   prob_x=sum(diag(prob_k)*prob_x_k);                                       % p(x|w)
   for k=1:K
      prob_k_x(k,:)=prob_k(k)*prob_x_k(k,:)./prob_x;                        % p(k|x_n) = gamma_{nk}
   end,
   y=diag(1./sum(prob_k_x'))*prob_k_x*E_train ;                              % y(k) = mu_k
   
   dist=sum((y.*y)')'*ones(1,Ntrain) + E2train -2*y*E_train';   
   sig2=(1/D)*diag(1./sum(prob_k_x'))*(sum((dist.*prob_k_x)')')+ 10e-5;
   sig_arr(:,t)=sig2;
   prob_k=sum(prob_k_x')/Ntrain;                                            % pi_k
   Etrain_arr(t)=gm_cost2(E_train,y,sig2,prob_k);
   Etest_arr(t)=gm_cost2(E_test,y,sig2,prob_k);
   if Plot_boleen
   % plot centers
   if rem(t,5)==0,
     figure(1)
     for k=1:K,
        h_ite = plot(y(k,1),y(k,2),'r*'); text(y(k,1),y(k,2),[int2str(k),'-',int2str(t)] );
        drawnow;
     end
     figure(2), 
     subplot(2,1,1),plot(sig_arr'),title('Convergence of variance parameters'), drawnow
     subplot(2,1,2),plot(1:t,Etrain_arr,'b'),hold on,plot(1:t,Etest_arr,'r'),
     hold off, title('Training (blue) and Test (red) errors '),
     drawnow
   end
   end
end   %end EM
if Plot_boleen
    figure(2), legend([h_train, h_test,h_init,h_ite],'train','test','\mu_k^{init}','\mu_k^{ite}')

    figure()
    labels = {'e1', 'e2'};
    num_plot = 2;
    counter = 1;
    for i = 1:num_plot
        for j = 1:num_plot
            subplot(num_plot,num_plot,counter)
            plot(E_train(:,i),E_train(:,j),'.'),hold on,
            for k=1:K,
               plot(y(k,i),y(k,j),'r*'),
               plot(y(k,i)+sqrt(sig2(k))*sin(2*pi*(0:31)/30),   y(k,j)+sqrt(sig2(k))*cos(2*pi*(0:31)/30),'g')
            end        
            xlabel(labels{i})
            ylabel(labels{j})
            counter = counter + 1;        
        end
    end
end
frac_worst = 0.03;
num_window = size(E,1);
[E_sum_train prob_train]=gm_cost(E_train,y,sig2,prob_k);
[E_sum_test prob_test]=gm_cost(E_test,y,sig2,prob_k);

E_prob = [prob_train prob_test];

E_prob_sort = sort(E_prob);
% ten_pro = E_prob_sort(floor(frac_worst*num_window));
% 
% [I,J] = find(ten_pro > E_prob);
% 
% ten_worst = zeros(1, num_window);
% ten_worst(E_prob <= ten_pro) = 1;
% ten_worst(E_prob > ten_pro) = 0;
% 
% Seizure_time = [94 123];
% figure()
% subplot(2,1,1)
% bar(Seizure_time(1):Seizure_time(2), ten_worst(Seizure_time(1):Seizure_time(2)), 'r','EdgeColor', 'r')
%         hold on 
%         bar(1:Seizure_time(1)-1, ten_worst(1:Seizure_time(1)-1), 'b','EdgeColor', 'b')
%         hold on
%         bar((Seizure_time(2)+1):num_window, ten_worst(Seizure_time(2)+1:end), 'b','EdgeColor', 'b')
%         legend('Seizure', 'Non-Seizure')
% title([num2str(frac_worst*100) '% most abnormal'])
% axis([0 num_window 0 1.2])
% subplot(2,1,2)
% histogram(J,floor(num_window/10))
% axis([0 num_window 0 8])
% title('Bin size: 10 sek MIXTURE')
end 