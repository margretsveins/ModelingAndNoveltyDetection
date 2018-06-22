% Define parameters 
e_max = 0.3;
t_alpha = 0.8;
alpha_0 = 0.6;

% Get data 
train = Y_train_F2{1,10}(1:4,:);
test = Y_test_F2{1,10}(1:4,:);
Seizure_time = seizure{1,10};
title_plot{1,10}

train_mean = mean(train,2); 
train_std = std(train')';
train = (train-train_mean)./train_std;
test = (test-train_mean)./train_std;

[K_vec, y, sig2, Mahalanobis_dist_vec_train, Q_t_vec] = model2_new_cooling(train, 'e_max', e_max, 't_alpha', t_alpha, 'aplpha_0', alpha_0);


%% Plots 
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
plot(Mahalanobis_dist_vec_test)
hold on 
plot(repmat(Q_t_max, 1,length(Mahalanobis_dist_vec_test)))
xlabel('t')
ylabel('min mahala')
title('Test')

subplot(1,2,2)
plot(Mahalanobis_dist_vec_train)
hold on 
plot(repmat(Q_t_max, 1,length(Mahalanobis_dist_vec_train)))
xlabel('t')
ylabel('min mahala')
title('Train')

% K 
figure()
plot(K_vec)
xlabel('t')
ylabel('#K')

% Log plot 
ten_worst = zeros(1,length(test));
ten_worst(outlier_index) = 1; 
sum(ten_worst(Seizure_time))/length(Seizure_time)

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

e_max = [0.1, 0.2, 0.5];
y_cell = {};
sig2_cell = {};
figure()
for i = 1:length(e_max)
    [K_vec, y, sig2] = model2_new_cooling(train,test, 'e_max', e_max(i), 't_alpha', t_alpha, 'aplpha_0', alpha_0);
    y_cell{1,i} = y;
    sig2_cell{1,i} = sig2;
    plot(K_vec)
    hold on 
end 
ylabel('K_t')
xlabel('iterations (t)')
legend('e_{max}: 0.1', 'e_{max}: 0.2', 'e_{max}: 0.5')

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
    h1 = plot(repmat(Q_t_max, 1,length(Mahalanobis_dist_vec_test)))
    legend([h1], {['e_{max}: ' num2str(e_max(i))]})
    axis([0 3600 1 15]);
    xlabel('Time (s)')
    ylabel('Mahalanobis distance')
end

%% Log plot for many iter 

Q_t_max = 2*log(1./e_max);
iter_vec = [1 10 25 50 75 100];
 clear Outlier_matrix

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
    Outlier_matrix(q,:) = ten_worst;
end 

figure()
for i = 1:6
    subplot(3,2,i)
    Outliers = Outlier_matrix(1:iter_vec(i),:);
    if i > 1        
        ten_worst_mean = mean(Outliers);
%         ten_worst_mean(ten_worst_mean > 0) = 1;
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
e_max = [0.1 0.2 0.4];
t_alpha = 1;
alpha_0 = 0.7;

TPR_subject_cum = [];
FPR_subject_cum = [];
K_subject = {};
sig2_mean = {}; 
sig2_std = {}; 
Outlier_subject = {};

iter = 25;
for d = 1:length(Y_train_F2)
    % Get data 
    opt_num_pc = num_pc(d);
    train = Y_train_F2{1,d}(1:opt_num_pc,:);
    test = Y_test_F2{1,d}(1:opt_num_pc,:);
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
    for e = 1:length(e_max)
        Q_t_max = 2*log(1./e_max(e));
        Outlier_matrix = [];
        for q = 1:iter
            [K_vec, y, sig2] = model2_new_cooling(train,test, 'e_max', e_max(e), 't_alpha', t_alpha, 'aplpha_0', alpha_0);
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
    end
%     Outlier_subject{1,d} = Outlier_matrix;
    TPR_subject_cum(d,:) = TPR_cum;
    FPR_subject_cum(d,:) = FPR_cum;
    sig2_mean{1,d} = sig2_mean_iter;
    sig2_std{1,d} = sig2_std_iter;
    K_subject{1,d} = K_iter;       
end


%% Bar plot wtih TPR / FPR for different e_max
figure()
subplot(2,1,1)
% TPR_bar = [TPR_subject_cum{1,1}(25,:);TPR_subject_cum{1,2}(25,:)];
bar(TPR_subject_cum)
legend('\alpha_0 = 0.3', '\alpha_0 = 0.5', '\alpha_0 = 0.7', '\alpha_0 = 0.9')
ylabel('Sensitivity')
xlabel('Subject')

subplot(2,1,2)
% FPR_bar = [FPR_subject_cum{1,1}(25,:);FPR_subject_cum{1,2}(25,:)];
bar(1 -FPR_subject_cum)
legend('\alpha_0 = 0.3', '\alpha_0 = 0.5', '\alpha_0 = 0.7', '\alpha_0 = 0.9')
ylabel('Specificity')
xlabel('Subject')

K_mean = [];
for d = 1:10
    K_mean = [K_mean ;mean(K_subject{1,d})];
end 

figure 
bar(K_mean)
legend('\alpha_0 = 0.3', '\alpha_0 = 0.5', '\alpha_0 = 0.7', '\alpha_0 = 0.9')

%% 
n = length(TPR_subject_cum);
mean_sensitivity = mean(TPR_subject_cum);
mean_specificity = mean(1 -FPR_subject_cum);
std_sensitivity = std(TPR_subject_cum);
std_specificity = std(1 -FPR_subject_cum);

lower_sensitivity = mean_sensitivity - std_sensitivity/sqrt(n);
upper_sensitivity = mean_sensitivity + std_sensitivity/sqrt(n);
lower_specificity = mean_specificity - std_specificity/sqrt(n);
upper_specificity = mean_specificity + std_specificity/sqrt(n);

figure()
x_axis = 70:5:95;
h1 = plot(x_axis,mean_sensitivity, 'r')
hold on 
plot(x_axis,lower_sensitivity, 'r--')
hold on 
plot(x_axis,upper_sensitivity, 'r--')
hold on 

h2 = plot(x_axis,mean_specificity, 'b')
hold on 
plot(x_axis,lower_specificity, 'b--')
hold on 
plot(x_axis,upper_specificity, 'b--')
legend([h1 h2], {'Sensitivity', 'Specificity'},'FontSize',30)
% xticks(1:length(var_target))
axis([70 95 0 1])
xlabel('Variance explained','FontSize',30,'FontWeight','bold')
ylabel('%','FontSize',34,'FontWeight','bold')
yticks(0:0.1:1)
yticklabels(0:10:100)
xtickformat('percentage')
set(gca, 'FontSize', 30)



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
%% Optimal variance
% Define parameters 
var_target = [70 75 80 85 90 95];
t_alpha = 1;
alpha_0 = 0.7;
e_max = 0.2;
Q_t_max = 2*log(1./e_max);

TPR_subject_cum = [];
FPR_subject_cum = [];
K_subject = {};
sig2_mean = {}; 
sig2_std = {}; 
Outlier_subject = {};

iter = 25;
for d = 1:length(Y_train_F2)
    d
    K_iter = [];
    TPR_cum = [];
    FPR_cum = [];
    for e = 1:length(var_target)   
        var_target(e)
         % Get data 
        var_explained = var_explained_F2{1,d};
        num_prin = 1;
        while sum(var_explained(1:num_prin)) < var_target(e)
            num_prin = num_prin + 1;
        end 
        
        train = Y_train_F2{1,d}(1:num_prin,:);
        test = Y_test_F2{1,d}(1:num_prin,:);
        Seizure_time = seizure{1,d};

        % Standardise data
        train_mean = mean(train,2); 
        train_std = std(train')';
        train = (train-train_mean)./train_std;
        test = (test-train_mean)./train_std;   


        sig2_mean_iter = [];
        sig2_std_iter = [];     
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
    end
%     Outlier_subject{1,d} = Outlier_matrix;
    TPR_subject_cum(d,:) = TPR_cum;
    FPR_subject_cum(d,:) = FPR_cum;
    sig2_mean{1,d} = sig2_mean_iter;
    sig2_std{1,d} = sig2_std_iter;
    K_subject{1,d} = K_iter;       
end


%% 
Outlier_matrix_total = Outlier_subject1;
for d = 1:length(Y_train_F2)    
    Seizure_time = seizure{1,d};
    Outlier_matrix = Outlier_matrix_total{d,1};
    new_growth = [];
    for q = 1:25
        if q == 1
            ten_worst = Outlier_matrix(1,:);
            outliers = ten_worst;
        else
            ten_worst = mean(Outlier_matrix(1:q,:)) > 0;
            iter = Outlier_matrix(q,:);
            outliers = outliers + iter;
            new_outliers = sum(outliers == 1);
            new_growth = [new_growth new_outliers];
        end
        true_positive = sum(ten_worst(Seizure_time));
        False_postive = sum(ten_worst)-true_positive;
        num_seizure = length(Seizure_time);
        Condition_negative = length(test)-num_seizure;

        TPR = true_positive/num_seizure;
        FPR = False_postive/Condition_negative;        

        TPR_cum(d,q) = TPR;
        FPR_cum(d,q) = FPR;
        
    end
    growth(d,:) = new_growth;
end

mean_TPR_iter = mean(TPR_cum);
lower_TPR_iter = mean_TPR_iter - std(TPR_cum)/sqrt(n);
upper_TPR_iter = mean_TPR_iter +std(TPR_cum)/sqrt(n);

mean_FPR_iter = mean(1 -FPR_cum);
lower_FPR_iter = mean_FPR_iter - std(1 - FPR_cum)/sqrt(n);
upper_FPR_iter = mean_FPR_iter +std(1 - FPR_cum)/sqrt(n);

average_growth = mean(growth);
lower_growth = average_growth - std(growth)/sqrt(n);
upper_growth = average_growth +std(growth)/sqrt(n);

%%
figure()

subplot(2,2,1)
    h1 = plot(mean_TPR_iter, 'b')
    hold on 
    plot(lower_TPR_iter, 'b--')
    hold on 
    plot(upper_TPR_iter, 'b--')
    legend('Sensitivity')
    xlabel('Number of runs','FontSize',20,'FontWeight','bold')
    ylabel('%','FontSize',20,'FontWeight','bold')
    yticks(0:0.1:1)
    yticklabels(0:10:100)
    axis([1 25 0.4 1])
    set(gca, 'FontSize', 20)
subplot(2,2,2)
    h2 = plot(mean_FPR_iter, 'b')
    hold on 
    plot(lower_FPR_iter, 'b--')
    hold on 
    plot(upper_FPR_iter, 'b--')
    legend('Specificity')
    xlabel('Number of runs','FontSize',20,'FontWeight','bold')
    ylabel('%','FontSize',20,'FontWeight','bold')
    axis([1 25 0.4 1])
    ylabel('%')
    yticks(0:0.1:1)
    yticklabels(0:10:100)
    set(gca, 'FontSize', 20)
subplot(2,1,2)
plot(average_growth, 'b')
hold on 
plot(lower_growth, 'b--')
hold on 
plot(upper_growth, 'b--')
xlabel('Number of runs','FontSize',20,'FontWeight','bold')
ylabel('Number of new novel vectors','FontSize',20,'FontWeight','bold')
title('Average growth of novel vectors','FontSize',20,'FontWeight','bold')
set(gca, 'FontSize', 20)
