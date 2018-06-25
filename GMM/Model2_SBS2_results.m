% Define parameters 
e_max = 0.3;
t_alpha = 6.6;
alpha_0 = 0.4;

% Get data 
var_explained = var_explained;
num_prin = 1;
while sum(var_explained(1:num_prin)) < var_target
    num_prin = num_prin + 1;
end 

train = Y_train{1,d}(1:num_prin,:);

% Standardize data 
train_mean = mean(train,2); 
train_std = std(train')';
train = (train-train_mean)./train_std;

Y_test_NON_stand = [];
for i = 1:length(Y_test_non)
    test = Y_test_non{1,i}(1:num_prin,:);
    test = (test-train_mean)./train_std;
    
    Y_test_NON_stand = [Y_test_NON_stand test];
end

Y_test_WITH_stand = [];
for i = 1:length(Y_test_with)
    test = Y_test_with{1,i}(1:num_prin,:);
    test = (test-train_mean)./train_std;    
    Y_test_WITH_stand = [Y_test_WITH_stand test];
end

[K_vec, y, sig2] = model2_new_cooling(train, 'e_max', e_max, 't_alpha', t_alpha, 'aplpha_0', alpha_0);

%% Plots 
Q_t_max = 2*log(1./e_max)
outlier_index_NON = [];
outlier_index_WITH = [];
outlier_index_TRAIN = [];
clear Mahalanobis_dist_vec_test Mahalanobis_dist_vec_train
for i = 1:length(Y_test_NON_stand)
    x_t = Y_test_NON_stand(:,i);
    Mahalanobis_dist = diag((x_t - y')'*(x_t - y'))./sig2;   
    Mahalanobis_dist = min(Mahalanobis_dist);
    Mahalanobis_dist_vec_test_NON(i) = Mahalanobis_dist;
    if Mahalanobis_dist > Q_t_max
        outlier_index_NON = [outlier_index_NON i];
    end 
end 

for i = 1:length(Y_test_WITH_stand)
    x_t = Y_test_WITH_stand(:,i);
    Mahalanobis_dist = diag((x_t - y')'*(x_t - y'))./sig2;   
    Mahalanobis_dist = min(Mahalanobis_dist);
    Mahalanobis_dist_vec_test_WITH(i) = Mahalanobis_dist;
    if Mahalanobis_dist > Q_t_max
        outlier_index_WITH = [outlier_index_WITH i];
    end 
end 

for i = 1:length(train)
    x_t = train(:,i);
    Mahalanobis_dist = diag((x_t - y')'*(x_t - y'))./sig2;   
    Mahalanobis_dist = min(Mahalanobis_dist);
    Mahalanobis_dist_vec_train(i) = Mahalanobis_dist;
    if Mahalanobis_dist > Q_t_max
        outlier_index_TRAIN = [outlier_index_TRAIN i];
    end 
end 

num_WITH = length(outlier_index_WITH);
num_NON = length(outlier_index_NON);

% Mahalanobis dist for both train and test 
figure()
subplot(3,1,1)
plot(Mahalanobis_dist_vec_train)
hold on 
plot(repmat(Q_t_max, 1,length(Mahalanobis_dist_vec_train)),'LineWidth', 1.5)
xlabel('Time (s)', 'FontSize', 16)
ylabel('Smallest Mahalanobis distance', 'FontSize', 14)
title('Training data')
yticks(Q_t_max)
yticklabels('Q_t(\epsilon_{max})')
axis([0 length(Mahalanobis_dist_vec_train) 0 5*Q_t_max])

subplot(3,1,2)
plot(Mahalanobis_dist_vec_test_NON)
hold on 
plot(repmat(Q_t_max, 1,length(Mahalanobis_dist_vec_test_NON)),'LineWidth', 1.5)
xlabel('Time (s)', 'FontSize', 16)
ylabel('Smallest Mahalanobis distance', 'FontSize', 14)
title(['Test data - NON, num novel: ' num2str(num_NON)])
yticks(Q_t_max)
yticklabels('Q_t(\epsilon_{max})')
axis([0 length(Mahalanobis_dist_vec_test_NON) 0 5*Q_t_max])

subplot(3,1,3)
plot(Mahalanobis_dist_vec_test_WITH)
hold on 
plot(repmat(Q_t_max, 1,length(Mahalanobis_dist_vec_test_WITH)),'LineWidth', 1.5)
xlabel('Time (s)', 'FontSize', 16)
ylabel('Smallest Mahalanobis distance', 'FontSize', 14)
title(['Test data - WITH, , num novel: ' num2str(num_WITH)])
yticks(Q_t_max)
yticklabels('Q_t(\epsilon_{max})')
axis([0 length(Mahalanobis_dist_vec_test_WITH) 0 5*Q_t_max])


%%
% K 
figure()
plot(K_vec)
xlabel('t')
ylabel('#K')

% Log plot 
ten_worst = zeros(1,length(test));
ten_worst(outlier_index) = 1; 

figure()

    b1 = bar(ten_worst, 'b','EdgeColor', 'b', 'EdgeAlpha', 0.05)       
    hold on
    bar(Seizure_time, ten_worst(Seizure_time), 'r','EdgeColor', 'r')        
    legend('Non-Seizure','Seizure')
    alpha(b1, 0.5)    
    b1.EdgeAlpha = 0.10
hold on 
% title(['Most abnormal datapoints(' num2str(Q_t_max) '%), or outliers, for ' Plot_title])
axis([0 length(ten_worst) 0 1.2])
xlabel('Time (s)')
yticks([1])
yticklabels('Outliers')