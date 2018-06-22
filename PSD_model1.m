
nwin = [];
nfft = nwin;
num_sec = 1;
sample_freq = 256;
length_window = sample_freq*num_sec;
noverlap = nwin * 0.5;

 j = 1;
 Y_matrix = {};
for k = 1:length(Data)
    data = Data{1,k};
    num_window = floor(size(data,2)/length_window);
    E_matrix = zeros(6*21,num_window);
    j = 1;
    for i = 1:num_window
        % Get "window" of the data to summerize
        eeg_data = data(:,j:j+length_window-1)';  
        % Get the features
        E = getFeatures(eeg_data, nwin, sample_freq, length_window);
        E_matrix(:,i) = E;
        j = j+length_window;
    end
    
%     size_train = 0.2;
%     % Standardize data with mean and std from train data 
%     E_train = E_matrix(:,1:num_window*size_train)';
%     E_test = E_matrix(:,num_window*size_train+1:end)';
%     
%     num_obs_train = size(E_train,1);
%     num_obs_test = size(E_test,1);
%     E_train_mean = mean(E_train);
%     E_train_std = repmat(std(E_train),num_obs_train,1);
%     E_test_std = repmat(std(E_train),num_obs_test,1);
%     E_train_normal = (E_train - E_train_mean) ./ E_train_std;
%     E_test_normal = (E_test - E_train_mean) ./ E_test_std;
%     
%     % Run PCA on training data and applied to test data 
%     [U, Y_trans, latent, tsquared, var_explained] = pca(E_train_normal, 'Centered', false);
%     Y_train = Y_trans';
%     Y_test = U'*E_test_normal';
%     Y_full = [Y_train, Y_test];
%     Y_matrix{1,k}= Y_full;
%     var_explained_matrix(k,:) = var_explained';
    
    [U, Y_trans, latent, tsquared, var_explained] = pca(E_matrix', 'Centered', false);
    Y_trans = Y_trans';
    Y_matrix{1,k}= Y_trans;
    var_explained_matrix(k,:) = var_explained';
end 

%% Print ROC plot for different number of variance explained
clear AreaROC_matrix AreaROC_matrix_stand Num_prin_matrix
var_target = [75:1:99];
for i =1:length(Data)
    seizure_before = seizure{1,i};
    seiz_start = floor(seizure_before(1)/num_sec);
    seiz_end = ceil(seizure_before(end)/num_sec);
    seizure_new = seiz_start:seiz_end;
    [FPR_matrix, TPR_matrix, AreaROC,num_prin_vec] = ROC_PC_spectral(Y_matrix{1,i}, var_explained_matrix(i,:), length_window,0.2,var_target, seizure_new, title_plot{1,i}, plot_name{1,i});
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
var_target = 99;
for i = 1:length(title_plot)
    seizure_before = seizure{1,i};
    seiz_start = floor(seizure_before(1)/num_sec);
    seiz_end = ceil(seizure_before(end)/num_sec);
    seizure_new = seiz_start:seiz_end;
    [FPR, TPR, AreaROC,num_prin_vec] = ROC_PC_spectral(Y_matrix{1,i}, var_explained_matrix(i,:), length_window,0.2,var_target, seizure_new, title_plot{1,i}, plot_name{1,i});
    FPR_matrix{i,:} =  FPR; 
    TPR_matrix{i,:} = TPR;
    AreaROC_matrix_99(i) = AreaROC;
    num_comp_99(i) = num_prin_vec;
end
AreaROC_matrix_99 = AreaROC_matrix_99';
num_comp_99 = num_comp_99';
%% 
figure()
for i = 1:6
    plot(FPR_matrix{i,:}, TPR_matrix{i,:})
    hold on
end 
plot(FPR_matrix{i,:},FPR_matrix{i,:},'--')
xlabel('FPR')
ylabel('TPR')
title('99% of variance explained')
legend('Patient 2', 'Patient 4', 'Patient 5', 'Patient 7', 'Patient 10','Patient 13', 'Location', 'southeast')

figure()
for i = 7:11
    plot(FPR_matrix{i,:}, TPR_matrix{i,:})
    hold on
end
plot(FPR_matrix{i,:},FPR_matrix{i,:},'--')
xlabel('FPR')
ylabel('TPR')
title('99% of variance explained')
legend('Patient 14', 'Patient 16', 'Patient 20', 'Patient 21', 'Patient 22', 'Location', 'southeast')

%% Log normal

var_target = 99;
for i =1:length(Y_matrix)
    Y_full = Y_matrix{1,i};
    var_explained = var_explained_matrix(i,:);    
    num_prin = 1;
    while sum(var_explained(1:num_prin)) < var_target
        num_prin = num_prin + 1;
    end 
    Y = Y_full(1:num_prin,:);
    Seizure_test = seizure{1,i};
    SeizStart = floor(Seizure_test(1)/num_sec);
    SeizEnd = ceil(Seizure_test(end)/num_sec);
    frac_worst = 0.05;       
    num_window = size(Y,2);
    size_train = 0.2;
    E_train = Y(:,1:size_train*num_window);
    E_test = Y(:,size_train*num_window+1:end);
    [log_p, log_p_train, log_p_test, ten_worst] = log_normal(E_train, Y, 'Plot', true, title_plot{1,i}, plot_name{1,i}, frac_worst, 'Seizure', true, [SeizStart SeizEnd], 10);
    sum(ten_worst(SeizStart:SeizEnd))/length(Seizure_test)
end

%% Optimal threshold

% To find the optimal threshold we define objective function as the
% distance from the ROC curve to the point (0,1) that is 
%           C(t) = sqrt(1 - TPR(t)^2  + FPR(t)^2)

clear FPR_matrix TPR_matrix
var_target = 99;
for i = 1:length(title_plot)
    [FPR, TPR, AreaROC,num_prin_vec, f] = ROC_PC_spectral(Y_matrix{1,i}, var_explained_matrix(i,:), length_window,var_target, seizure{1,i}, title_plot{1,i}, plot_name{1,i});
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

%% Gaussian mixture inspection 
num_prin = 5;
size_train = 0.1:0.1:0.9;
for k = 1:length(Data)
Y_full = Y_matrix{1,k };
Y_full = Y_full(1:num_prin,:);
num_window = size(Y_full,2);

    for i = 1:length(size_train) 

        num_train = num_window*size_train(i);

        Y_train = Y_full(:,1:num_train);
        Y_test = Y_full(:,num_train+1:end);

        mu_train = mean(Y_train');
        mu_train_matrix(:,i) = mu_train';
        std_train = std(Y_train');

        % Lets find 95% confidence interval
        Z = 1.960;
        n_train = length(Y_train);
        conf_int_train = Z * std_train/sqrt(n_train);
        conf_int_train_matrix(:,i) = conf_int_train;
        mu_test = mean(Y_test');
        mu_test_matrix(:,i) = mu_test;
        std_test = std(Y_test');

        % Lets find 95% confidence interval
        Z = 1.960;
        n_test = length(Y_test);
        conf_int_test = Z * std_test/sqrt(n_test);
        conf_int_test_matrix(:,i) = conf_int_test;

        Sigma = cov(Y_train');   
        subplot(3,3,i)
        heatmap(Sigma)
        title(['Training size: ' num2str(size_train(i)) ' P' num2str(k)])  

    end 


    true_mean = mean(Y_full');
    true_mean_matrix(:,k) = true_mean';
    
    figure() 
    subplot(2,1,1)
    errorbar(mu_train_matrix', 2*conf_int_train_matrix')
    legend('P1', 'P2', 'P3', 'P4', 'P5') 
    title(['Mean train for P' num2str(k)])
    subplot(2,1,2)
    errorbar(mu_test_matrix', 2*conf_int_test_matrix')
    legend('P1', 'P2', 'P3', 'P4', 'P5') 
    title(['Mean test for P' num2str(k)])
    xlabel('size of train')

    figure()
    for i = 1:num_prin
    subplot(3,2,i)
    plot(mu_train_matrix(i,:))
    hold on 
    errorbar(mu_test_matrix(i,:),  2*conf_int_test_matrix(i,:))
    hold on 
    plot(repmat(true_mean(i),9,1))
    legend('Mean of train data', 'Mean of test data', '"true" mean', 'Location', 'northeast')
    xlabel('Size of train')
    title(['Component ' num2str(i) ' for P' num2str(k)])
    end 
    
    figure()
    for i = 1:num_prin
    plot(cumsum(Y_full(i,:))./(1:num_window))
    hold on 
    end 
    legend('C1','C2','C3','C4','C5')
end 

%% 
num_train = 3599;

Y_train = Y_full(:,1:num_train);
Y_test = Y_full(:,num_train+1:end);

mu_test = mean(Y_test,2)
mu_train = mean(Y_train,2)
mu_full = mean(Y_full,2)

sum_mu_full = sum(Y_full(1,:))
sum_mu_train = sum(Y_train(1,:))

Y_full(1,3600)

%% 

% figure()
% plot(Y_full(1,3500:3600))

figure()
for i = 1:5
plot(cumsum(Y_full(i,:))./(1:3600))
hold on 
end 
legend('C1','C2','C3','C4','C5')

%% 
num_prin = 6
for k = 1:length(Data) 
    clear mu_window diff_window SigmaMu SigmaMu_matrix
    data = Y_matrix{1,k}; 
    data = data(1:num_prin,:);
    num_window = length(data);
    mu_window = cumsum(data)./(1:num_window);
    diff_window = data - mu_window;
    Seizure_time = seizure{1,k};
    
 
    SigmaMu = [];
    size_train = 0.1:0.1:0.9;
    for q = 1:length(size_train)
        num_train_window = floor(num_window*size_train(q));
        Sigma = cov(data(:,1:num_train_window)');
        mu = mean(data(:,1:num_train_window)');
        for i = 3:num_window
%             Sigma = cov(data(:,1:i)');
%             mu = mu_window(:,i);
            median_window = median(data(:,1:i)')';
            median_matrix(:,i) = median_window;
            SigmaMu(i) = 0.5*((data(:,i)-mu')'*(Sigma\(data(:,i)-mu')));
        end
        SigmaMu_matrix(q,:) = SigmaMu;
    end 
    
    figure()
    for i = 1:num_prin
        subplot(3,2,i)
        plot(mu_window(i,3:end)', '--')
        hold on 
        plot(median_matrix(i,:)')
        title(['Mean for different num of window Patient ' num2str(k) ' comp ' num2str(i)])
    end 
    
    figure()
    for i = 1:num_prin 
        subplot(3,2,i)
        diff_plot = diff_window(i,:)';
        plot(diff_plot)
        hold on 
        plot(Seizure_time,diff_plot(Seizure_time), 'r');
        title(['x-mu Patient ' num2str(k) 'comp ' num2str(i)])
    end 
    
    figure()
    for i = 1:length(size_train)
        subplot(3,3,i)
            SigmaMu_plot = SigmaMu_matrix(i,:);
            plot(SigmaMu_plot)
            hold on 
            plot(Seizure_time,SigmaMu_plot(Seizure_time), 'r');
            title(['(x-mu)^T Sigma(x-mu) Patient ' num2str(k) ' size train ' num2str(size_train(i))])
    end
    
    figure()
    for i = 1:num_prin 
        subplot(3,2,i)
        data_plot = data(i,:)';
        plot(data_plot)
        hold on 
        plot(Seizure_time,data_plot(Seizure_time), 'r');
        title(['Patient ' num2str(k) 'comp ' num2str(i)])
    end 
    

end 


%% 
for k = 1:length(Data) 
    clear mu_window diff_window SigmaMu SigmaMu_matrix MedianMad_matrix
    data = Y_matrix{1,k}; 
    data = data(1:num_prin,:);
    num_window = length(data);
    mu_window = cumsum(data)./(1:num_window);
    diff_window = data - mu_window;
    Seizure_time = seizure{1,k};
    
 
    SigmaMu = [];
    MedianMad = [];
    size_train = 0.1:0.1:0.9;
    for q = 1:length(size_train)
        num_train_window = floor(num_window*size_train(q));
        data_train = data(:,1:num_train_window);
        Sigma = cov(data_train');
        mu = mean(data(:,1:num_train_window)');
        M = median(data_train,2);
        MAD = median(abs((data_train - M)),2);
        MAD_matrix = diag(MAD);
        Sigma_matrix(:,:,q) = Sigma;
        MAD_matrix_plot(:,:,q) = MAD_matrix;
        for i = 1:num_window
            median_window = median(data(:,1:i)')';
            median_matrix(:,i) = median_window;
            SigmaMu(i) = 0.5*((data(:,i)-mu')'*(Sigma\(data(:,i)-mu')));            
            MedianMad(i) = 0.5*((data(:,i)-M)'*inv(MAD_matrix)*(data(:,i)-M));
        end
        SigmaMu_matrix(q,:) = SigmaMu;
        MedianMad_matrix(q,:) = MedianMad;
    end
    
    
    c1 = 1;
    c2 = 2; 
    c3 = 3;
    c4 = 4;
     figure()
    for i = 1:5
        subplot(5,4,c1)
        MedianMad_plot = MedianMad_matrix(i,:);
        h2 = plot(MedianMad_plot, '-')
        hold on             
        plot(Seizure_time,MedianMad_plot(Seizure_time), 'r');
        title(['Median Patient ' num2str(k) ' TrainSize: ' num2str(size_train(i))])        
        
        subplot(5,4,c2)
        imagesc(MAD_matrix_plot(:,:,i),[min(min(MAD_matrix_plot(:,:,i))) max(max(MAD_matrix_plot(:,:,i)))]), colormap('gray'), colorbar
        title(['MAD matrix ' num2str(k) ' TrainSize: ' num2str(size_train(i))])   
        
        subplot(5,4,c3)
        SigmaMu_plot = SigmaMu_matrix(i,:);
        h1 = plot(SigmaMu_plot, '-')
        hold on 
        plot(Seizure_time,SigmaMu_plot(Seizure_time), 'r');
        title(['Mean Patient ' num2str(k) ' TrainSize: ' num2str(size_train(i))])  
        
  
        subplot(5,4,c4)
        imagesc(Sigma_matrix(:,:,i),[min(min(Sigma_matrix(:,:,i))) max(max(Sigma_matrix(:,:,i)))]), colormap('gray'), colorbar
        title(['Sigma ' num2str(k) ' TrainSize: ' num2str(size_train(i))])  
        c1 = c1 + 4;
        c2 =  c2 + 4;
        c3 =  c3 + 4;
        c4 =  c4 + 4;
    end
    
    c1 = 1;
    c2 = 2; 
    c3 = 3;
    c4 = 4;
     figure()
    for i = 6:9
        subplot(4,4,c1)
        MedianMad_plot = MedianMad_matrix(i,:);
        h2 = plot(MedianMad_plot, '-')
        hold on             
        plot(Seizure_time,MedianMad_plot(Seizure_time), 'r');
        title(['Median Patient ' num2str(k) ' TrainSize: ' num2str(size_train(i))])        
        
        subplot(4,4,c2)
        imagesc(MAD_matrix_plot(:,:,i),[min(min(MAD_matrix_plot(:,:,i))) max(max(MAD_matrix_plot(:,:,i)))]), colormap('gray'), colorbar
        title(['MAD matrix ' num2str(k) ' TrainSize: ' num2str(size_train(i))])   
        
        subplot(4,4,c3)
        SigmaMu_plot = SigmaMu_matrix(i,:);
        h1 = plot(SigmaMu_plot, '-')
        hold on 
        plot(Seizure_time,SigmaMu_plot(Seizure_time), 'r');
        title(['Mean Patient ' num2str(k) ' TrainSize: ' num2str(size_train(i))])  
        
  
        subplot(4,4,c4)
        imagesc(Sigma_matrix(:,:,i),[min(min(Sigma_matrix(:,:,i))) max(max(Sigma_matrix(:,:,i)))]), colormap('gray'), colorbar
        title(['Sigma ' num2str(k) ' TrainSize: ' num2str(size_train(i))])  
        c1 = c1 + 4;
        c2 =  c2 + 4;
        c3 =  c3 + 4;
        c4 =  c4 + 4;
    end
    
   
end    

%% 
%% Print ROC plot for optimal variance explained for all the patient 
clear FPR_matrix TPR_matrix Mean_AUC
var_target = 99;
size_train = 0.1:0.1:0.9;
for q = 1:length(size_train)
    for i = 1:length(title_plot)
        seizure_before = seizure{1,i};
        seiz_start = floor(seizure_before(1)/num_sec);
        seiz_end = ceil(seizure_before(end)/num_sec);
        seizure_new = seiz_start:seiz_end;
        [FPR, TPR, AreaROC,num_prin_vec] = ROC_PC_spectral(Y_matrix{1,i}, var_explained_matrix(i,:), length_window,size_train(q),var_target, seizure_new, title_plot{1,i}, plot_name{1,i});
        FPR_matrix{i,:} =  FPR; 
        TPR_matrix{i,:} = TPR;
        AreaROC_matrix_99(i) = AreaROC;
        num_comp_99(i) = num_prin_vec;
    end
    Mean_AUC(:,q) = AreaROC_matrix_99;
end 

figure()
for i = 1:length(Data)
    subplot(2,5,i)
    plot(Mean_AUC(i,:))
    xlabel('Size train')
    ylabel('AUC')
    title(title_plot{1,i})
end 


%% Learning curve
clear error_train error_test
xAxis_jump = 10;
size_train = 0.1:0.1:0.9;
% size_train = 0.2;
var_target = 99;
figure()
for j = 1:length(Data)
    num_prin = 1;
    Data_full = Y_matrix{1,j};
    var_explained = var_explained_matrix(j,:);
    while sum(var_explained(1:num_prin)) < var_target
        num_prin = num_prin + 1;
    end 

    for i = 1:length(size_train)       
        Data_pc = Data_full(1:num_prin,:);
        Data_train_pc = Data_pc(:,1:floor(num_window*size_train(i)));
        [log_p, log_p_train, log_p_test] = log_normal(Data_train_pc, Data_pc); 
        error_train(i) =abs(mean(log_p_train));
        error_test(i) = abs(mean(log_p_test));
        
        Z = 1.960;
        n_train = length(log_p_train);
        std_train = std(log_p_train);
        conf_int_train(i) = Z * std_train/sqrt(n_train);
        
        n_test = length(log_p_test);
        std_test = std(log_p_test);
        conf_int_test(i) = Z * std_test/sqrt(n_test);
      
    end
    subplot(2,5,j)
    Model_1_errorPlot(error_train, error_test, conf_int_train, conf_int_test,size_train(i), xAxis_jump, title_plot{1,j}, plot_name{1,j})
end
%% 
for q = 1:length(Y_matrix)
    Y = Y_matrix{1,q};
    mean_comp(:,q) = mean(Y,2);
    std_comp = std(Y,[],2);
    median_comp(:,q) = median(Y,2);
    Z = 1.960;
    n = length(Y);
    conf_int(:,q) = Z .* std_comp./sqrt(n);     
end 

num_comp = 6;
figure()
for i = 1:num_comp
    subplot(3,2,i)
    errorbar(mean_comp(i,:),2*conf_int(i,:))
    hold on
    plot(median_comp(i,:))
    legend('mean', 'median')
    xlabel('patient')
    title(['Compenent ' num2str(i)])
end

%% 
figure()
errorbar(mean_comp(1,:),2*conf_int(1,:))
hold on
plot(median_comp(1,:))
legend('mean', 'median')
xlabel('patient')
title(['Compenent ' num2str(i)])

