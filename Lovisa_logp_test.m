warning off
clear
close all
randn('state',0)   %fix random generator seed

%%%%%%%%%%%%%%%%%%%%%%% Parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
w_t = [1 2 0.5]';	 % True weights
noiselevel = 0.75;   % Standard deviation of Gaussian noise on data
d = size(w_t,1);     % Number of dimensions
Nmin=5;              % Minimal training set size
Nmax=5000000;             % Maximal training set size
Ntest= 100000;        % Size of test set 
repetitions=10;      % number of repetitions


    % d-dimensional model data set
    X1test=randn(Ntest,d);
    X1test(:,1)=ones(Ntest,1);
    % Compute the truth for the test set
    Ttest=(X1test*w_t);
    noisetest = randn(Ntest,1) * noiselevel;
    % Add noise
    Ttest= Ttest + noisetest;
    % Small model (d-1) dimensional
    X2test=X1test(:,1:(d-1));
    XX1=randn(Nmax,d);
    XX1(:,1)=ones(Nmax,1);
    TT = (XX1*w_t);
    noise = randn(Nmax,1) * noiselevel;
    TT = TT + noise;
    XX2=XX1(:,1:(d-1));
    
   XX1 = XX1';
   X1test = X1test';
   

size_train = 0.1:0.01:0.9;
for j = 1:length(size_train)  
    Y_train = XX1(2:3, floor(size_train(j)*Nmax));
    Y_full = [Y_train X1test(2:3,:)];

    % Get the log likelihood
    [log_p, log_p_train, log_p_test] = log_normal(Y_train, Y_full); 
    log_p_test_vec(j) = median(abs(log_p_test));
    log_p_train_vec(j) = median(abs(log_p_train));
end 

figure()
semilogy(log_p_test_vec)
hold on 
semilogy(log_p_train_vec)
legend('Train', 'Test')

%% 
figure()
hist(XX1(2:3,:)')
hold on 
hist(X1test(2:3,:)')

%% 

train = XX1(2:3,:);
Sigma = cov(train');
mu = mean(train,2);

test = mu + [0.001; 0];

log_p = -0.5*3*log(det(Sigma)) - 0.5*((test-mu)'*inv(Sigma)*(test-mu))

%% 
% Get a random training data wih some noise
Ntrain = 1000;
Ntest = 500;
noise_train=0.001; 
noise_test=10; 
[xtrain,xtest]=getdata(Ntrain,Ntest,noise_train,noise_test);
xtrain = xtrain';
xtest = xtest';

figure()
subplot(2,2,1)
hist(xtrain(1,:))
title('Train comp1')
subplot(2,2,2)
hist(xtrain(2,:))
title('Train comp2')
subplot(2,2,3)
hist(xtest(1,:))
title('Test comp1')
subplot(2,2,4)
hist(xtest(2,:))
title('Test comp2')

figure()
mu_vector = [];
size_train = 0.1:0.1:1;
for j = 1:length(size_train)  
    Y_train = xtrain(:, 1:floor(size_train(j)*Ntrain));
    Y_full = [Y_train xtest];
    
    num_window = length(Y_full);
    num_train = length(Y_train);
    
    mu = mean(Y_train,2);
    Sigma = cov(Y_train');    
    
    mu_vector(:,j) = mu;
    
    
    log_p = [];
    log_p_train = [];
    log_p_test = [];
    for i = 1:num_window
        factor(i) = - 0.5*((Y_full(:,i)-mu)'*(Sigma\(Y_full(:,i)-mu)));
        log_p(1,i) = -0.5*num_window*log(det(Sigma)) - 0.5*((Y_full(:,i)-mu)'*(Sigma\(Y_full(:,i)-mu)));
    end
    % Get the log likelihood
    log_p_train = log_p(1:num_train);
    log_p_test = log_p(num_train+1:end);
    
    subplot(2,5,j)
    plot(sort(log_p))
    hold on 
    plot(sort(log_p_train))
    hold on 
    plot(sort(log_p_test))
    title(['Log_p - Size of train ' num2str(size_train(j))])
    legend('Log_p','Log_p - Train','Log_p - Test')
    
    median_full(j) = median(log_p);
    median_train(j) = median(log_p_train);
    median_test(j) = median(log_p_test);
    
    mean_full(j) = mean(log_p);
    mean_train(j) = mean(log_p_train);
    mean_test(j) = mean(log_p_test);
end 

true_mu = mean(xtrain,2);
true_mu_vector = repmat(true_mu,1,length(size_train));
figure()
plot(mu_vector')
hold on 
plot(true_mu_vector', '--')

figure()
plot(median_full)
hold on 
plot(median_train)
hold on 
plot(median_test)
legend('full', 'train', 'test')

figure()
plot(mean_full)
hold on 
plot(mean_train)
hold on 
plot(mean_test)
legend('full', 'train', 'test')

