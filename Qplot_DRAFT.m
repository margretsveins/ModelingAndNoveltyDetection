% Test for q plots
% Figure of Q plot not standardized
figure()
threshold = -15;
for i = 1:length(log_p_train_F1)
    % Feature set 1 
    F1_log_p_train = log_p_train_F1{1,i};
    F1_log_p_test = log_p_test_F1{1,i};
    
%     mu_train_F1 = mean(F1_log_p_train);
%     std_train_F1 = std(F1_log_p_train);
%     F1_log_p_train = (F1_log_p_train-mu_train_F1)/std_train_F1;
    F1_train_thres(i) = sum(F1_log_p_train<threshold)/length(F1_log_p_train);
   

    F1_log_p_test = (F1_log_p_test-mu_train_F1)/std_train_F1;
    F1_test_thres(i) = sum(F1_log_p_test<threshold)/length(F1_log_p_test);

    
    [Q_train, Q_test,t_train, t_test] = QplotNew_MS(F1_log_p_train, F1_log_p_test, jump);
    subplot(2,2,1)
    hold on 
    plot(t_train,Q_train)
    title('F1 train')
    xlabel('\theta')
    ylabel('Q(\theta)')
    subplot(2,2,2)
    hold on 
    plot(t_test,Q_test)
    title('F1 test')
    xlabel('\theta')
    ylabel('Q(\theta)')

    % Feature set 2
    F2_log_p_train = log_p_train_F2{1,i};
    F2_log_p_test = log_p_test_F2{1,i};
    
%     mu_train_F2 = mean(F2_log_p_train);
%     std_train_F2 = std(F2_log_p_train);
%     F2_log_p_train = (F2_log_p_train-mu_train_F2)/std_train_F2;
    F2_train_thres(i) = sum(F2_log_p_train<threshold)/length(F2_log_p_train);
    
    F2_log_p_test = (F2_log_p_test-mu_train_F2)/std_train_F2;
    F2_test_thres(i) = sum(F2_log_p_test<threshold)/length(F2_log_p_test);
    
    [Q_train, Q_test,t_train, t_test] = QplotNew_MS(F2_log_p_train, F2_log_p_test, jump);
    subplot(2,2,3)
    hold on 
    plot(t_train,Q_train)
    title('F2 train')
    xlabel('\theta')
    ylabel('Q(\theta)')
    subplot(2,2,4)
    hold on
    plot(t_test,Q_test)
    title('F2 test')
    xlabel('\theta')
    ylabel('Q(\theta)')    
    
end

%%
threshold = -15;
figure()
for i = 1:length(log_p_train_F1)
    % Feature set 1 
    F1_log_p_train = log_p_train_F1{1,i};
    F1_log_p_test = log_p_test_F1{1,i};
    
%     mu_train_F1 = mean(F1_log_p_train);
%     std_train_F1 = std(F1_log_p_train);
%     F1_log_p_train = (F1_log_p_train-mu_train_F1)/std_train_F1;
    F1_train_thres(i) = sum(F1_log_p_train<threshold)/length(F1_log_p_train);
   

%     F1_log_p_test = (F1_log_p_test-mu_train_F1)/std_train_F1;
    F1_test_thres(i) = sum(F1_log_p_test<threshold)/length(F1_log_p_test);

    
    [Q_train, Q_test,t_train, t_test] = QplotNew(F1_log_p_train, F1_log_p_test, jump);
    hold on
    plot(t_train,Q_train)
    hold on
    title('F1 train')
    xlabel('\theta')
    ylabel('Q(\theta)')
    axis([-150 5 0 0.5]) 
    
end

figure()
for i = 1:length(log_p_train_F1)
    % Feature set 1 
    F1_log_p_train = log_p_train_F1{1,i};
    F1_log_p_test = log_p_test_F1{1,i};
    
%     mu_train_F1 = mean(F1_log_p_train);
%     std_train_F1 = std(F1_log_p_train);
%     F1_log_p_train = (F1_log_p_train-mu_train_F1)/std_train_F1;
    F1_train_thres(i) = sum(F1_log_p_train<threshold)/length(F1_log_p_train);
   

%     F1_log_p_test = (F1_log_p_test-mu_train_F1)/std_train_F1;
    F1_test_thres(i) = sum(F1_log_p_test<threshold)/length(F1_log_p_test);

    
    [Q_train, Q_test,t_train, t_test] = QplotNew(F1_log_p_train, F1_log_p_test, jump);
    hold on
    plot(t_test,Q_test)
    hold on
    title('F1 test')
    xlabel('\theta')
    ylabel('Q(\theta)')
    axis([-150 5 0 0.5])     
end

figure()
for i = 1:length(log_p_train_F1)
    % Feature set 2
    F2_log_p_train = log_p_train_F2{1,i};
    F2_log_p_test = log_p_test_F2{1,i};

    F2_train_thres(i) = sum(F2_log_p_train<threshold)/length(F2_log_p_train);
    F2_test_thres(i) = sum(F2_log_p_test<threshold)/length(F2_log_p_test);
    
    [Q_train, Q_test,t_train, t_test] = QplotNew(F2_log_p_train, F2_log_p_test, jump);
    hold on
    plot(t_train,Q_train)
    hold on
    title('F2 train')
    xlabel('\theta')
    ylabel('Q(\theta)')
    axis([-150 5 0 0.5])    

    
end

figure()
for i = 1:length(log_p_train_F1)
    % Feature set 2
    F2_log_p_train = log_p_train_F2{1,i};
    F2_log_p_test = log_p_test_F2{1,i};
    
%     mu_train_F2 = mean(F2_log_p_train);
%     std_train_F2 = std(F2_log_p_train);
%     F2_log_p_train = (F2_log_p_train-mu_train_F2)/std_train_F2;
    F2_train_thres(i) = sum(F2_log_p_train<threshold)/length(F2_log_p_train);
    
    F2_log_p_test = (F2_log_p_test-mu_train_F2)/std_train_F2;
    F2_test_thres(i) = sum(F2_log_p_test<threshold)/length(F2_log_p_test);
    
    [Q_train, Q_test,t_train, t_test] = QplotNew(F2_log_p_train, F2_log_p_test, jump);
    hold on
    plot(t_test,Q_test)
    hold on
    title('F2 test')
    xlabel('\theta')
    ylabel('Q(\theta)')  
    axis([-150 5 0 0.5])     
end


