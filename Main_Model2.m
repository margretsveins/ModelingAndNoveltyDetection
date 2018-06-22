% Define parameters 
e_max = 0.2;
t_alpha = 1;
alpha_0 = 0.7;

% Get data 
train = Y_train_F2{1,2}(1:10,:);
test = Y_test_F2{1,2}(1:10,:);
Seizure_time = seizure{1,2};
% start_seiz = floor(Seizure_time(1)/8);
% end_seiz = ceil(Seizure_time(end)/8);
% Seizure_time = start_seiz:end_seiz;
title_plot{1,2}

train_mean = mean(train,2); 
train_std = std(train')';
train = (train-train_mean)./train_std;
test = (test-train_mean)./train_std;

[K_vec, y, sig2, Mahalanobis_dist_vec_train, Q_t_vec] = model2_new_cooling(train, 'e_max', e_max, 't_alpha', t_alpha, 'aplpha_0', alpha_0);


%% Plots 
e_max = 0.2;
Q_t_max = 2*log(1./e_max)
outlier_index = []
clear Mahalanobis_dist_vec_test Mahalanobis_dist_vec_train
for i = 1:length(test)
    x_t = test(:,i);
    Mahalanobis_dist = diag((x_t - y')'*(x_t - y'))./sig2;   
    Mahalanobis_dist = min(Mahalanobis_dist);
    Mahalanobis_dist_vec_test(i) = Mahalanobis_dist;
    if Mahalanobis_dist > Q_t_max
        outlier_index = [outlier_index i];
    end 
end 

for i = 1:length(train)
    x_t = train(:,i);
    Mahalanobis_dist = diag((x_t - y')'*(x_t - y'))./sig2;   
    Mahalanobis_dist = min(Mahalanobis_dist);
    Mahalanobis_dist_vec_train(i) = Mahalanobis_dist;
    if Mahalanobis_dist > Q_t_max
        outlier_index = [outlier_index i];
    end 
end 

% Mahalanobis dist for both train and test 
figure()
subplot(1,2,1)
plot(Mahalanobis_dist_vec_train)
hold on 
plot(repmat(Q_t_max, 1,length(Mahalanobis_dist_vec_train)),'LineWidth', 1.5)
xlabel('Time (s)', 'FontSize', 16)
ylabel('Smallest Mahalanobis distance', 'FontSize', 14)
title('Training data')
yticks(Q_t_max)
yticklabels('Q_t(\epsilon_{max})')
axis([0 3600 0 50])

subplot(1,2,2)
plot(Mahalanobis_dist_vec_test)
hold on 
plot(repmat(Q_t_max, 1,length(Mahalanobis_dist_vec_test)),'LineWidth', 1.5)
xlabel('Time (s)', 'FontSize', 16)
ylabel('Smallest Mahalanobis distance', 'FontSize', 14)
title('Test data')
yticks(Q_t_max)
yticklabels('Q_t(\epsilon_{max})')
axis([0 3600 0 50])
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

%% Plot K vec and test mahananobis dist for different e_max

e_max = [0.1 0.4 0.7];
y_cell = {};
sig2_cell = {};
figure()
for i = 1:length(e_max)
    [K_vec, y, sig2] = model2_new_cooling(train,test, 'e_max', e_max(i), 't_alpha', t_alpha, 'aplpha_0', alpha_0);
    y_cell{1,i} = y;
    sig2_cell{1,i} = sig2;
    plot(K_vec, 'LineWidth', 2)
    hold on 
end 
ylabel('K_t', 'FontSize', 16)
xlabel('iterations (t)', 'FontSize', 16)
legend('e_{max}: 0.1', 'e_{max}: 0.4', 'e_{max}: 0.7')
set(gca, 'FontSize', 16)

figure()
for i = 1:length(e_max)
    y = y_cell{1,i};
    sig2 = sig2_cell{1,i};
    for t = 1:length(test)
        x_t = test(:,t);
        Mahalanobis_dist = diag((x_t - y')'*(x_t - y'))./sig2;   
        Mahalanobis_dist = min(Mahalanobis_dist);
        Mahalanobis_dist_vec_test(t) = Mahalanobis_dist;
    end 
    Q_t_max = 2*log(1./e_max(i));
    subplot(3,1,i)
    plot(Mahalanobis_dist_vec_test)
    hold on 
    h1 = plot(repmat(Q_t_max, 1,length(Mahalanobis_dist_vec_test)),'LineWidth', 1.5)
    legend([h1], {['e_{max}: ' num2str(e_max(i))]})
    axis([0 3600 1 15]);
    xlabel('Time (s)','FontSize', 16)
    ylabel('Mahalanobis distance','FontSize', 16)
end

%% Log plot for many iter 

Q_t_max = 2*log(1./e_max);
iter_vec = [1 10 25 50 75 100];
 clear Outlier_matrix
e_max = 0.2;
opt_num_pc = num_pc(2);
train = Y_train_F2{1,2}(1:opt_num_pc,:);
test = Y_test_F2{1,2}(1:opt_num_pc,:);
Seizure_time = seizure{1,2};

% Standardise data
train_mean = mean(train,2); 
train_std = std(train')';
train = (train-train_mean)./train_std;
test = (test-train_mean)./train_std;  
clear Mahalanobis_dist_vec_test Mahalanobis_dist_vec_train  
for q = 1:100
    [K_vec, y, sig2] = model2_new_cooling(train,test, 'e_max', e_max, 't_alpha', t_alpha, 'aplpha_0', alpha_0);
    outlier_index = [];
    for i = 1:length(test)
        x_t = test(:,i);
        Mahalanobis_dist = diag((x_t - y')'*(x_t - y'))./sig2;   
        Mahalanobis_dist = min(Mahalanobis_dist);
        Mahalanobis_dist_vec_test(i) = Mahalanobis_dist;
        if Mahalanobis_dist > Q_t_max
            outlier_index = [outlier_index i];
        end 
    end 
    ten_worst = zeros(1,length(test));
    ten_worst(outlier_index) = 1;
    Outlier_matrix_p21(q,:) = ten_worst;
end 

figure()
for i = 1:6
    subplot(3,2,i)
    Outliers = Outlier_matrix_p21(1:iter_vec(i),:);
    if i > 1        
        ten_worst_mean = mean(Outliers);
        ten_worst_mean(ten_worst_mean > 0) = 1;
    else
        ten_worst_mean = Outliers;
    end 
    num_outlier = sum(ten_worst_mean>0);
    num_seizure = sum(ten_worst_mean(Seizure_time)>0);
    b1 = bar(ten_worst_mean, 'b','EdgeColor', 'b', 'EdgeAlpha', 0.05)       
    hold on
    bar(Seizure_time, ten_worst_mean(Seizure_time), 'r','EdgeColor', 'r')        
    legend('Non-Seizure','Seizure')
    alpha(b1, 0.5)    
    b1.EdgeAlpha = 0.10
    hold on 
    title([title_plot{1,9} ' - ' num2str(num_outlier) ' outeliers, ' num2str(num_seizure) ' seizure points,  after ' num2str(iter_vec(i)) ' iterations'])
    axis([0 length(ten_worst_mean) 0 1.2])
    xlabel('Time (s)')
end 

%% TPR, FPR, K_vec for each patient 
% Define parameters 
e_max = 0.4;
t_alpha = 100;
% alpha_0= [0.3 0.5 0.7 0.9 1.5 1.9];
alpha_0= [30 50 70];

TPR_subject_cum = [];
FPR_subject_cum = [];
K_subject = {};
sig2_mean = {}; 
sig2_std = {}; 
Outlier_subject = {};

var_target = 95; 

iter = 1;
for d = 1:1%length(Y_train_F2)
    % Get data     
    var_explained = var_explained_F2{1,d};
    num_prin = 1;
    while sum(var_explained(1:num_prin)) < var_target
        num_prin = num_prin + 1;
    end 
    
    train = Y_train_F2{1,d}(1:2,:);
    test = Y_test_F2{1,d}(1:2,:);
%     train = Y_train_F2{1,d}(1:num_prin,:);
%     test = Y_test_F2{1,d}(1:num_prin,:);
    Seizure_time = seizure{1,d};

    % Standardise data
    train_mean = mean(train,2); 
    train_std = std(train')';
    train = (train-train_mean)./train_std;
    test = (test-train_mean)./train_std;   

    K_iter = [];
    TPR_cum = [];
    FPR_cum = [];
    sig2_mean_iter = [];
    sig2_std_iter = [];
    for e = 1:length(alpha_0)
        d
        alpha_0(e)
        Q_t_max = 2*log(1./e_max);
        Outlier_matrix = [];
        for q = 1:iter
            q
            [K_vec, y, sig2, Mahalanobis_dist_vec, Q_t_vec, y_cell, sig2_cell, prob_k_cell] = model2_new_cooling(train,test, 'e_max', e_max, 't_alpha', t_alpha, 'aplpha_0', alpha_0(e));
            outlier_index = [];
            for i = 1:length(test)
                x_t = test(:,i);
                Mahalanobis_dist = diag((x_t - y')'*(x_t - y'))./sig2;   
                Mahalanobis_dist = min(Mahalanobis_dist);
                if Mahalanobis_dist > Q_t_max
                    outlier_index = [outlier_index i];
                end 
            end 
            ten_worst = zeros(1,length(test));
            ten_worst(outlier_index) = 1;
            Outlier_matrix(q,:) = ten_worst; 
            K_iter(q,e) = K_vec(end);
            sig2_mean_iter(q,e) = mean(sig2);
            sig2_std_iter(q,e) = std(sig2);            
        end 
        Outlier_subject{d,e} = Outlier_matrix;
        
        % Compute for TPR and FPR for "total outlier" 
        ten_worst = zeros(1,length(test));
        if q == 1             
            ten_worst(outlier_index) = 1;
        else
            ten_worst(mean(Outlier_matrix) > 0) = 1;
        end
        true_positive = sum(ten_worst(Seizure_time));
        False_postive = sum(ten_worst)-true_positive;
        num_seizure = length(Seizure_time);
        Condition_negative = length(test)-num_seizure;

        TPR = true_positive/num_seizure;
        FPR = False_postive/Condition_negative;        

        TPR_cum(1,e) = TPR;
        FPR_cum(1,e) = FPR;
        y_change{1,e} = y_cell;
        sig2_change{1,e} = sig2_cell;
        prob_k_change{1,e} = prob_k_cell;
    end
%     Outlier_subject{1,d} = Outlier_matrix;
    TPR_subject_cum(d,:) = TPR_cum;
    FPR_subject_cum(d,:) = FPR_cum;
    sig2_mean{1,d} = sig2_mean_iter;
    sig2_std{1,d} = sig2_std_iter;
    K_subject{1,d} = K_iter;       
end



%% Bar plot wtih TPR / FPR for different e_max
% subjects = [02 05 07 10 13 14 16 20 21 22]; 
figure()
subplot(2,1,1)
% TPR_bar = [TPR_subject_cum{1,1}(25,:);TPR_subject_cum{1,2}(25,:)];
bar(TPR_subject_cum)
legend('alpha_0: 0.1', 'alpha_0: 0.3','alpha_0: 0.5','alpha_0: 0.7', 'alpha_0: 0.9')
ylabel('Sensitivity')
% xlabel('Subject')
xticks(1:10)
% xticklabels(subjects)
axis([0 11 0 1])
subplot(2,1,2)
% FPR_bar = [FPR_subject_cum{1,1}(25,:);FPR_subject_cum{1,2}(25,:)];
bar(1 -FPR_subject_cum)
 legend('alpha_0: 0.1', 'alpha_0: 0.3','alpha_0: 0.5','alpha_0: 0.7', 'alpha_0: 0.9')
ylabel('Specificity')
% xlabel('Subject')
xticks(1:10)
% xticklabels(subjects)
axis([0 11 0 1])

figure()
bar((TPR_subject_cum + (1 -FPR_subject_cum))/2)

K_mean = [];
for d = 1:10
    K_mean = [K_mean ;mean(K_subject{1,d})];
end 

for i = 1:10 
    for e = 1:length(alpha_0) 
        num_outlier(i,e) = sum(mean(Outlier_subject_emax{i,e})>0)/length(Data{1,i});
    end 
end 

figure 
subplot(2,1,1)
bar(K_mean)
legend('alpha_0: 0.1', 'alpha_0: 0.3','alpha_0: 0.5','alpha_0: 0.7', 'alpha_0: 0.9')
ylabel('Number of clusters, k')
xlabel('Subject')
xticks(1:10)
% xticklabels(subjects)
subplot(2,1,2)
bar(num_outlier)
ylabel('Number of outliers/Number of training points')
xlabel('Subject')
legend('alpha_0: 0.1', 'alpha_0: 0.3','alpha_0: 0.5','alpha_0: 0.7', 'alpha_0: 0.9')
xticks(1:10)
% xticklabels(subjects)

%% 
clear TPR_cum FPR_cum TPR_matrix FPR_matrix
for i = 1:10
    OutlierMatrix = Outlier_subject{i,2};
    Seizure_time = seizure{1,i};
    for j = 1:25
        if j == 1             
            ten_worst = OutlierMatrix(1,:);
        else
            ten_worst(mean(OutlierMatrix) > 0) = 1;
        end
        true_positive = sum(ten_worst(Seizure_time));
        False_postive = sum(ten_worst)-true_positive;
        num_seizure = length(Seizure_time);
        Condition_negative = length(test)-num_seizure;

        TPR = true_positive/num_seizure;
        FPR = False_postive/Condition_negative;        

        TPR_cum(1,j) = TPR;
        FPR_cum(1,j) = FPR;        
    end 
    TPR_matrix(i,:) = TPR_cum;
    FPR_matrix(i,:) = FPR_cum;
end 


%% 
Z = 1.960;
n = size(TPR_matrix,1);

mean_TPR = mean(TPR_matrix);
std_TPR = std(TPR_matrix)

conf_int = Z * std_TPR/sqrt(n);
lower_TPR = mean_TPR - conf_int;
upper_TPR = mean_TPR + conf_int;

mean_FPR = mean(FPR_matrix);
std_FPR = std(FPR_matrix)

conf_int = Z * std_FPR/sqrt(n);
lower_FPR = mean_FPR - conf_int;
upper_FPR = mean_FPR + conf_int;

figure()
subplot(2,1,1)
plot(mean_TPR)
hold on 
plot(lower_TPR, '--')
hold on 
plot(upper_TPR, '--')

subplot(2,1,2)
plot(mean_FPR)
hold on 
plot(lower_FPR, '--')
hold on 
plot(upper_FPR, '--')

%% 
figure()
iter_vec = [1 10 15 25];
for i = 1:4
    Seizure_time = seizure{1,3};
    Outlier_matrix = Outlier_subject{1,1};
    subplot(2,2,i)
    Outliers = Outlier_matrix(1:iter_vec(i),:);
    if i > 1        
        ten_worst_mean = mean(Outliers);
        ten_worst_mean(ten_worst_mean > 0) = 1;
    else
        ten_worst_mean = Outliers;
    end 
    num_outlier = sum(ten_worst_mean>0);
    num_seizure = sum(ten_worst_mean(Seizure_time)>0);
    b1 = bar(ten_worst_mean, 'b','EdgeColor', 'b', 'EdgeAlpha', 0.05)       
    hold on
    bar(Seizure_time, ten_worst_mean(Seizure_time), 'r','EdgeColor', 'r')        
    legend('Non-Seizure','Seizure')
    alpha(b1, 0.5)    
    b1.EdgeAlpha = 0.10
    hold on 
    title([title_plot{1,3} ' - ' num2str(num_outlier) ' outeliers, ' num2str(num_seizure) ' seizure points,  after ' num2str(iter_vec(i)) ' iterations'])
    axis([0 length(ten_worst_mean) 0 1.2])
    xlabel('Time (s)')
end 

%% 
Outlier_matrix_ = Outlier_subject{1,1};
Outlier_matrix_p21_5PC = Outlier_matrix_p21(1:25,:);
Test{1,1} = Outlier_matrix_;
Test{1,2} = Outlier_matrix_p21_5PC;

sum(Outlier_matrix_')
sum(Outlier_matrix_p21_5PC')
clear TPR_matrix FPR_matrix TPR_cum FPR_cum

    OutlierMatrix = Outlier_matrix_p21;
    Seizure_time = seizure{1,2};
    for j = 1:100
        if j == 1             
            ten_worst = OutlierMatrix(1,:);
        else
            ten_worst(mean(OutlierMatrix(1:j,:)) > 0) = 1;
        end
        true_positive = sum(ten_worst(Seizure_time));
        False_postive = sum(ten_worst)-true_positive;
        num_seizure = length(Seizure_time);
        Condition_negative = length(test)-num_seizure;

        TPR = true_positive/num_seizure;
        FPR = False_postive/Condition_negative;        

        TPR_cum(1,j) = TPR;
        FPR_cum(1,j) = FPR;        
    end 
    TPR_matrix = TPR_cum;
    FPR_matrix= FPR_cum;


figure()
subplot(2,1,1)
plot(TPR_matrix)
title('TPR')
legend('PC: 6', 'PC:5')
axis([0 100 0.5 1])
subplot(2,1,2)
plot(FPR_matrix)
title('FPR')
legend('PC: 6', 'PC:5')
axis([0 100 0 0.2])
%% 
figure()
for i = 1:10
    var_explained = var_explained_F2{1,i}
    plot(cumsum(var_explained))    
    hold on 
end 
axis([0 126 0 100])
legend('2','5','7','10','13','14','16','20','21','22')

%% LOGPLOT for optimal value
iter = 5;


for d = 1:length(Y_train_F2)  
    h = figure()
    % Get data     
    var_explained = var_explained_F2{1,d};
    num_prin = 1;
    while sum(var_explained(1:num_prin)) < var_target
        num_prin = num_prin + 1;
    end 
    
    train = Y_train_F2{1,d}(1:num_prin,:);
    test = Y_test_clean_F2{1,d}(1:num_prin,:);
    Seizure_time = seizure{1,d};
    start_seiz = floor(Seizure_time(1)/8);
    end_seiz = ceil(Seizure_time(end)/8);
    Seizure_time = start_seiz:end_seiz;
    
    % Standardise data
    train_mean = mean(train,2); 
    train_std = std(train')';
    train = (train-train_mean)./train_std;
    test = (test-train_mean)./train_std;   
    
     e_max = 0.2;
     t_alpha = 1;
     alpha_0 = 0.7;

    Q_t_max = 2*log(1./e_max);
    Outlier_matrix = [];
    for q = 1:iter
        [K_vec, y, sig2] = model2_new_cooling(train,test, 'e_max', e_max, 't_alpha', t_alpha, 'aplpha_0', alpha_0);
        outlier_index = [];
        for i = 1:length(test)
            x_t = test(:,i);
            Mahalanobis_dist = diag((x_t - y')'*(x_t - y'))./sig2;   
            Mahalanobis_dist = min(Mahalanobis_dist);
            if Mahalanobis_dist > Q_t_max
                outlier_index = [outlier_index i];
            end 
        end 
        ten_worst = zeros(1,length(test));
        ten_worst(outlier_index) = 1;
        Outlier_matrix(q,:) = ten_worst;            
    end    
     
    ten_worst_mean = mean(Outlier_matrix);
    ten_worst_mean(ten_worst_mean > 0) = 1;

    num_outlier = sum(ten_worst_mean>0);
    num_seizure = sum(ten_worst_mean(Seizure_time)>0);
    b1 = bar(ten_worst_mean, 'b','EdgeColor', 'b', 'EdgeAlpha', 0.05)       
    hold on
    bar(Seizure_time, ten_worst_mean(Seizure_time), 'r','EdgeColor', 'r')        
    legend('Non-Seizure','Seizure')
    alpha(b1, 0.5)    
    b1.EdgeAlpha = 0.10
    title([title_plot{1,d} ' - ' num2str(num_outlier) ' outeliers, ' num2str(num_seizure) ' seizure points'])
    axis([0 length(ten_worst_mean) 0 1.2])
    xlabel('Time (s)')
    
%     saveas(h, sprintf('LogPlot_model2_subject_%s', plot_name{1,d}),'epsc')
    
end

