%% Get features and standardize 
% https://dsp.stackexchange.com/questions/23689/what-is-spectral-entropy
% file:///C:/Users/s161286/Downloads/00709563.pdf
nwin = [];
nfft = nwin;
num_sec = 1;
sample_freq = 256;
length_window = sample_freq*num_sec;
noverlap = nwin * 0.5;
num_chan = 21;

% GET TEST DATA
 j = 1;
 Feature_cell = {};
 Feature_norm_cell = {};
 PSE_cell = {};
for k = 1:length(Data)
    data = Data{1,k};
    num_window = floor(size(data,2)/length_window);
    Feature_matrix = zeros(6*num_chan,num_window);
    Feature_matrix_norm = zeros(5*num_chan,num_window);
    PSE_matrix = zeros(num_chan,num_window);
    j = 1;
    for i = 1:num_window
        % Get "window" of the data to summerize
        eeg_data = data(:,j:j+length_window-1)';  
        % Get the features
        [E, E_normal, PSE] = getFeatures(eeg_data, nwin, sample_freq, length_window, num_chan);
        Feature_matrix(:,i) = E;
        Feature_matrix_norm(:,i) = E_normal;
        PSE_matrix(:,i) = PSE;
        j = j+length_window;
    end
    
    Feature_cell{1,k} = Feature_matrix;
    Feature_norm_cell{1,k} = Feature_matrix_norm;
    PSE_cell{1,k} = PSE_matrix; 
    
end    

% GET CLEAN DATA
  j = 1;
 Feature_cell_train = {};
 Feature_norm_cell_train = {};
 PSE_cell_train = {};
for k = 1:length(Data_pre)
    data = Data_pre{1,k};
    num_window = floor(size(data,2)/length_window);
    Feature_matrix = zeros(6*num_chan,num_window);
    Feature_matrix_norm = zeros(5*num_chan,num_window);
    PSE_matrix = zeros(num_chan,num_window);
    j = 1;
    for i = 1:num_window
        % Get "window" of the data to summerize
        eeg_data = data(:,j:j+length_window-1)';  
        % Get the features
        [E, E_normal, PSE] = getFeatures(eeg_data, nwin, sample_freq, length_window, num_chan);
        Feature_matrix(:,i) = E;
        Feature_matrix_norm(:,i) = E_normal;
        PSE_matrix(:,i) = PSE;
        j = j+length_window;
    end
    
    Feature_cell_train{1,k} = Feature_matrix;
    Feature_norm_cell_train{1,k} = Feature_matrix_norm;
    PSE_cell_train{1,k} = PSE_matrix;    
end 

%% 
for k = 1:length(Feature_cell)
    PSE_matrix = PSE_cell{1,k};
    h1 = figure()
    for i = 1:num_chan
        subplot(7,3,i)
        semilogy(PSE_matrix(i,:))
        xlabel('Time (s)')
        ylabel('PSE')
        title(['Channel ' num2str(i) ' P' num2str(k)])
    end 
    saveas(h1, sprintf('PSE_features_%s', plot_name{1,k}),'epsc')
    
    features = Feature_cell{1,k};
    h2 = figure()
    j = 1;
    for i = 1:5:105
        features_chan = features(i:i+4,:);
        subplot(7,3,j)
        semilogy(features_chan')
        xlabel('Time (s)')
        ylabel('PSD')
        title(['Channel ' num2str(j) ' P' num2str(k)])
        j = j+1;
%         legend('Freq: [0-3[','Freq: [3,5[','Freq: [5,10[ ','Freq:  [10,21[','Freq: [21,44[' )
    end 
    saveas(h2, sprintf('PSD_features_%s', plot_name{1,k}),'epsc')
end 


%% 

clear Train_data_full random_matrix AreaROC train_error test_error mean_AUC train_error_matrix test_error_matrix AUC_matrix

% Make a train data set from all the patients
Train_data_full = [];
for i = 1:length(Feature_cell_train)
    Train_data_full =  [Train_data_full Feature_cell_train{1,i}];
end 

var_target = 95;
size_train = 0.1:0.05:0.9;
num_train_window = size(Train_data_full,2);
iter = 15;

% Make a random matrix to use for the iterations
random_matrix = zeros(iter,length(Train_data_full));
for p = 1:iter 
    random_matrix(p,:) = randsample(num_train_window, num_train_window);
end 

% Loop for  test data set 
for k = 1:length(Feature_cell)
    % Define test_data
    Test_data = Feature_cell{1,k};
    seizure_data = seizure{1,k};
    % Loop for size train
    for j = 1:length(size_train)
        random_matrix_size_train = random_matrix(:,1:floor(size_train(j)*num_train_window));       
        for q = 1:iter
            % Make a random vector to get a random sample from train data 
            random_vector = random_matrix_size_train(q,:);
            Train_data = Train_data_full(:,random_vector);
            
            % Standardize data 
            mu_train = mean(Train_data,2);
            std_train = std(Train_data')';
            
            Train_data_stand = (Train_data - mu_train)./std_train;
            Test_data_stand = (Test_data - mu_train)./std_train;            
            
             % Run PCA on training data and applied to test data 
            [U, Y_trans, latent, tsquared, var_explained] = pca(Train_data_stand', 'Centered', false);
            Y_train = Y_trans';   
            
            % Get number of PC to get the whanted variance 
            num_prin = 1;
            while sum(var_explained(1:num_prin)) < var_target
            num_prin = num_prin + 1;
            end 
            Y_train = Y_train(1:num_prin,:);
            Y_test = U'*Test_data_stand;
            Y_test = Y_test(1:num_prin,:);
            Y_full = [Y_train, Y_test];    

            % Get the log likelihood
             [log_p, log_p_train, log_p_test] = log_normal(Y_train, Y_full); 
%             [log_p, log_p_train, log_p_test] = log_median(Y_train, Y_full); 
            log_p_test_vec(q) = mean(abs(log_p_test));
            log_p_train_vec(q) = mean(abs(log_p_train));

            % Get the AUC        
            [FPR, TPR, f] = ROC_points_NEW(log_p, log_p_test, seizure_data, true);
            AreaROC(q) = trapz(FPR,TPR);
        end 
        train_error(j) = mean(log_p_train_vec);
        test_error(j) = mean(log_p_test_vec);
        mean_AUC(j) = mean(AreaROC);
    end 
    train_error_matrix(k,:) = train_error;
    test_error_matrix(k,:) = test_error;
    AUC_matrix(k,:) = mean_AUC;
end
%% 
j = 1; 
jump = 1;
xlabel_tick = [];
xtick = [];
while j <= length(size_train)
    xlabel_tick = [xlabel_tick floor(num_train_window*size_train(j))];
    xtick = [xtick j];
    j = j + jump;
end 

for k = 1:length(Feature_cell)
    h = figure()    
    
    subplot(2,1,1)
    plot(AUC_matrix(k,:))
    xlabel('Number of training samples')
    ylabel('AUC')
    title(['Mean - mean AUC ' title_plot{1,k}]) 
    xticks(xtick)
    xticklabels(xlabel_tick)
    xtickangle(45)
    
    subplot(2,1,2)
    semilogy(train_error_matrix(k,:))
    hold on 
    semilogy(test_error_matrix(k,:))
    legend('Train','Test')
    xlabel('Number of training points')
    ylabel('mean(abs(log))')
    title(['Mean - mean log likelihood ' title_plot{1,k}])
    xticks(xtick)
    xticklabels(xlabel_tick)
    xtickangle(45)    
    saveas(h, sprintf('LearningCurve_Mean_Clean_%s', plot_name{1,k}),'epsc')
end
    
h1 = figure()
for k = 1:length(Feature_cell) 
    subplot(2,5,k)
    plot(AUC_matrix(k,:))
    xlabel('Size of train')
    ylabel('AUC')
    title(['Mean - AUC ' title_plot{1,k}]) 
end
saveas(h1, sprintf('AUC_total_mean_Clean_%s', plot_name{1,k}),'epsc')

h2 = figure()
for k = 1:length(Feature_cell)    
    subplot(2,5,k)
    semilogy(train_error_matrix(k,:))
    hold on 
    semilogy(test_error_matrix(k,:))
    legend('Train','Test')
    xlabel('Size of train')
    ylabel('mean(abs(log))')
    title(['Mean - ' title_plot{1,k}])
end 
saveas(h2, sprintf('LearningCurve_Total_Mean_Clean_%s', plot_name{1,k}),'epsc')



