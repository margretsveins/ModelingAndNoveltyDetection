
%% Get features

nwin = [];
nfft = nwin;
num_sec = 1;
sample_freq = 256;
length_window = sample_freq*num_sec;
noverlap = nwin * 0.5;

 j = 1;
Y_train_cell = {};
Y_test_cell = {};
for k = 1:length(Data)
    % train data 
    Data_train = Data_pre{1,k};
    %Data_train = Data{1,k};
    num_window_train = floor(size(Data_train,2)/length_window);
    E_train = zeros(6*21,num_window_train);
    j = 1;
    for i = 1:num_window_train
        % Get "window" of the data to summerize
        eeg_data = Data_train(:,j:j+length_window-1)';  
        % Get the features
        E = getFeatures(eeg_data, nwin, sample_freq, length_window);
        E_train(:,i) = E;
        j = j+length_window;
    end
    
    % Get test data  
    Data_test = Data{1,k};
    num_window_test = floor(size(Data_test,2)/length_window);
    E_test = zeros(6*21,num_window_test);
    j = 1;
    for i = 1:num_window_test
        % Get "window" of the data to summerize
        eeg_data = Data_test(:,j:j+length_window-1)';  
        % Get the features
        E = getFeatures(eeg_data, nwin, sample_freq, length_window);
        E_test(:,i) = E;
        j = j+length_window;
    end
    
    % Standardize data with mean and std from train data 
    E_train = E_train';
    E_test = E_test';
    
    num_obs_train = size(E_train,1);
    num_obs_test = size(E_test,1);
    E_train_mean = mean(E_train);
    E_train_std = repmat(std(E_train),num_obs_train,1);
    E_test_std = repmat(std(E_train),num_obs_test,1);
    E_train_normal = (E_train - E_train_mean) ./ E_train_std;
    E_test_normal = (E_test - E_train_mean) ./ E_test_std;
    
    % Run PCA on training data and applied to test data 
    [U, Y_trans, latent, tsquared, var_explained] = pca(E_train_normal, 'Centered', false);
    Y_train = Y_trans';
    Y_train_cell{1,k}= Y_train;
    Y_test = U'*E_test_normal';
    Y_test_cell{1,k} = Y_test;
    var_explained_matrix(k,:) = var_explained';
end 


 
%% Log normal

var_target = 95;
for i =1:length(Y_train_cell)
    % Get data 
    Y_train = Y_train_cell{1,i};
    Y_test = Y_test_cell{1,i};
    Y_full = [Y_train Y_test];
    
    % Get right number of PC
    var_explained = var_explained_matrix(i,:);    
    num_prin = 1;
    while sum(var_explained(1:num_prin)) < var_target
        num_prin = num_prin + 1;
    end     
    Y = Y_full(1:num_prin,:);
    Y_train = Y_train(1:num_prin,:);
    
    % "Fix" the location of the seizure
    length_train = length(Y_train);
    length_test = length(Y_test);
    Seizure_test = seizure{1,i};
    SeizStart = floor(Seizure_test(1)/num_sec);%+length_train;
    SeizEnd = ceil(Seizure_test(end)/num_sec);%+length_train;
    
    frac_worst = 0.05;       
    num_window = size(Y,2);
    size_train = length_train/(length_train+length_test);

    [log_p, log_p_train, log_p_test, ten_worst] = log_normal(Y_train, Y, 'Plot', true, title_plot{1,i}, plot_name{1,i}, frac_worst, 'Seizure', true, [SeizStart SeizEnd], 10);
%     figure()
%     loglog(sort(log_p_train))
%     hold on 
%     loglog(sort(log_p_test))
%     legend('Train', 'Test')
end

%% Print ROC plot for different number of variance explained
clear AreaROC_matrix AreaROC_matrix_stand Num_prin_matrix
var_target = 75:1:99;
for i =1:length(Data)
    seizure_before = seizure{1,i};
    seiz_start = floor(seizure_before(1)/num_sec);
    seiz_end = ceil(seizure_before(end)/num_sec);
    seizure_new = seiz_start:seiz_end;
    
    Y_train = Y_train_cell{1,i};
    Y_test = Y_test_cell{1,i};
    Data_full = [Y_train Y_test];
    
    length_train = length(Y_train);
    length_test = length(Y_test);
    size_train = length_train/(length_train+length_test);
    
    [FPR_matrix, TPR_matrix, AreaROC,num_prin_vec] = ROC_PC_spectral(Data_full, var_explained_matrix(i,:), length_window,size_train,var_target, seizure_new, title_plot{1,i}, plot_name{1,i});
    AreaROC_matrix(i,:) =  AreaROC;
    Num_prin_matrix(i,:) = num_prin_vec;
end

%% Print average Area under the curve

mean_AreaROC = mean(AreaROC_matrix);
std_AreaROC = std(AreaROC_matrix);

% Lets find 95% confidence interval
Z = 1.960;
n = length(Data);
conf_int = Z * std_AreaROC/sqrt(n);

lower_AreaROC = mean_AreaROC - conf_int;
upper_AreaROC = mean_AreaROC + conf_int;

mean_pc = mean(Num_prin_matrix);
std__pc = std(Num_prin_matrix);
conf_int = Z * std__pc/sqrt(n);

lower__pc =  mean_pc - conf_int;
upper__pc = mean_pc + conf_int;
% xlabel_tick = [80 85 90 95 99 100];
xlabel_tick = 75:1:100;

figure()
yyaxis left 
h1 = plot(xlabel_tick,mean_AreaROC)
% hold on 
% plot(xlabel_tick,median_AreaROC)
hold on 
plot(xlabel_tick,lower_AreaROC, '--')
hold on 
plot(xlabel_tick,upper_AreaROC, '--')
ylabel('Area under the curve')
hold on
yyaxis right 
h2 = plot(xlabel_tick,mean_pc)
hold on
plot(xlabel_tick,lower__pc, '--')
hold on
plot(xlabel_tick,upper__pc, '--')
xlabel('% of variance explained')
ylabel('#Principal compenents')
legend([h1 h2], {'Mean area under the curve','Mean number of PC used (2d axis)'}, 'Location', 'northwest')

%% Print ROC plot for optimal variance explained for all the patient 

clear FPR_matrix TPR_matrix
var_target = 95;
for i = 1:length(title_plot)
    seizure_before = seizure{1,i};
    seiz_start = floor(seizure_before(1)/num_sec);
    seiz_end = ceil(seizure_before(end)/num_sec);
    seizure_new = seiz_start:seiz_end;
    
    Y_train = Y_train_cell{1,i};
    Y_test = Y_test_cell{1,i};
    Data_full = [Y_train Y_test];
    
    length_train = length(Y_train);
    length_test = length(Y_test);
    size_train = length_train/(length_train+length_test);
    
    [FPR, TPR, AreaROC,num_prin_vec] = ROC_PC_spectral(Data_full, var_explained_matrix(i,:), length_window,size_train,var_target, seizure_new, title_plot{1,i}, plot_name{1,i});
    FPR_matrix{i,:} =  FPR; 
    TPR_matrix{i,:} = TPR;
    AreaROC_matrix_95(i) = AreaROC;
    num_comp_95(i) = num_prin_vec;
end
AreaROC_matrix_95 = AreaROC_matrix_95';
num_comp_95 = num_comp_95';
%% 
figure()
for i = 1:6
    plot(FPR_matrix{i,:}, TPR_matrix{i,:})
    hold on
end 
plot(FPR_matrix{i,:},FPR_matrix{i,:},'--')
xlabel('FPR')
ylabel('TPR')
title('95% of variance explained')
legend('Patient 2', 'Patient 4', 'Patient 5', 'Patient 7', 'Patient 10','Patient 13', 'Location', 'southeast')

figure()
for i = 7:11
    plot(FPR_matrix{i,:}, TPR_matrix{i,:})
    hold on
end
plot(FPR_matrix{i,:},FPR_matrix{i,:},'--')
xlabel('FPR')
ylabel('TPR')
title('95% of variance explained')
legend('Patient 14', 'Patient 16', 'Patient 20', 'Patient 21', 'Patient 22', 'Location', 'southeast')

%% Optimal threshold

% To find the optimal threshold we define objective function as the
% distance from the ROC curve to the point (0,1) that is 
%           C(t) = sqrt(1 - TPR(t)^2  + FPR(t)^2)

clear FPR_matrix TPR_matrix
var_target = 95;
for i = 1:length(title_plot)
    seizure_before = seizure{1,i};
    seiz_start = floor(seizure_before(1)/num_sec);
    seiz_end = ceil(seizure_before(end)/num_sec);
    seizure_new = seiz_start:seiz_end;
    
    Y_train = Y_train_cell{1,i};
    Y_test = Y_test_cell{1,i};
    Data_full = [Y_train Y_test];
    
    length_train = length(Y_train);
    length_test = length(Y_test);
    size_train = length_train/(length_train+length_test);
    
    [FPR, TPR, AreaROC,num_prin_vec, f] = ROC_PC_spectral(Data_full, var_explained_matrix(i,:), length_window,size_train,var_target, seizure_new, title_plot{1,i}, plot_name{1,i});
    FPR_matrix2{i,:} =  FPR; 
    TPR_matrix2{i,:} = TPR;
    C1 = sqrt((1 - TPR).^2 + FPR.^2);
    C1_min = min(C1);
    C1_min_index = find(C1 == C1_min);
    
    C_ratio(i) = C1_min_index/length(FPR);
    optimal_threshold(i) = f(C1_min_index);
    
    Sensitivity(i) = TPR(C1_min_index);
    Specificity(i) = 1 - FPR(C1_min_index);
    h = figure()
    plot(FPR,TPR)
    hold on 
    opt = plot(FPR(C1_min_index), TPR(C1_min_index), '*')
    plot(FPR,FPR, '--')
    xlabel('FPR')
    ylabel('TPR')
    legend([opt], {'Optimal threshold'}, 'Location', 'southeast')
    title(title_plot{1,i})
    
%      saveas(h, sprintf('ROC_OptThres_%s', plot_name{1,i}),'epsc')            
end
Sensitivity = Sensitivity';
Specificity = Specificity';