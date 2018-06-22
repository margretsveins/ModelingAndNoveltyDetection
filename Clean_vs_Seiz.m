%% Q - Plot  Train clean
size_train = 0.2;
length_window = 256;
t_cell = {};
Q_train_matrix = [];
Q_test_matrix = [];
var_target = 90;
j = 1;
for i = 1:11
    Data_clean =   [Data_pre{1,i} Data{1,i}];
    Data_train = Data_pre{1,i};
    Data_test = Data{1,i};
    [E, E_train, var_explained] = E_matrix2(Data_clean, Data_train, Data_test, length_window,'Plot', false ,[], 'Standardize', true);
    [E_pc, E_train_pc] = num_pc(E, E_train, var_explained, var_target);
    [log_p, log_p_train, log_p_test] = log_normal(E_train_pc, E_pc); 
    [t Q_train, Q_test] = Qplot(log_p, log_p_train, log_p_test);
    Q_train_matrix(j,:) = Q_train;
    Q_test_matrix(j,:) = Q_test;  
    j = j + 1;        
end 

%% Print ROC plot for different number of variance explained
clear AreaROC_matrix AreaROC_matrix_stand Num_prin_matrix
% var_target = [80 85 90 95 99];
var_target = 90;
FPR_matrix_CleanTrain = {};
TPR_matrix_CleanTrain = {};
FPR_matrix_seiz = {};
TPR_matrix_seiz = {};
for i = 1:length(title_plot)
    Data_clean =   [Data_pre{1,i} Data{1,i}];
    Data_train = Data_pre{1,i};
    Data_test = Data{1,i};
    [FPR_matrix, TPR_matrix, AreaROC,num_prin_vec] = ROC_PC(Data_clean, true, Data_train, Data_test, length_window,var_target, seizure{1,i}, title_plot{1,i}, plot_name{1,i});
    AreaROC_matrix(i,:) =  AreaROC;
    Num_prin_matrix(i,:) = num_prin_vec;
    FPR_matrix_CleanTrain{1,i} = FPR_matrix;
    TPR_matrix_CleanTrain{1,i} = TPR_matrix;
    [FPR_matrix, TPR_matrix, AreaROC,num_prin_vec] = ROC_PC(Data_test, false, Data_train, Data_test, length_window,var_target, seizure{1,i}, title_plot{1,i}, plot_name{1,i});    
    FPR_matrix_seiz{1,i} = FPR_matrix;
    TPR_matrix_seiz{1,i} = TPR_matrix;
end
%% Print ROC plot for optimal variance explained for all the patient 
clear FPR_matrix TPR_matrix
var_target = 90;
figure()
for i = 1:length(title_plot)
    [FPR, TPR, AreaROC,num_prin_vec] = ROC_PC(Data{1,i}, length_window,var_target, seizure{1,i}, title_plot{1,i}, plot_name{1,i});
    FPR_matrix{i,:} =  FPR; 
    TPR_matrix{i,:} = TPR;
    
   plot(FPR,TPR)
   hold on
end
% legend([],{'Patient 2', 'Patient 4', 'Patient 5', 'Patient 7', 'Patient 10','Patient 13', 'Patient 14', 'Patient 16', 'Patient 20', 'Patient 21', 'Patient 22'})
plot(FPR,FPR, '--')
xlabel('FPR')
ylabel('TPR')
title('ROC plot with 90% of variance explained')

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
xlabel_tick = [80 85 90 95 99 100];

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

%% Compare training on seizure and on clean data 

h = figure()
for i = 1:length(FPR_matrix_seiz)
%     figure()
%     plot(FPR_matrix_CleanTrain{1,i}', TPR_matrix_CleanTrain{1,i}')
%     hold on 
%     plot(FPR_matrix_seiz{1,i}', TPR_matrix_seiz{1,i}')    
%     title(title_plot{1,i})
%     plot(FPR_matrix_seiz{1,i}',FPR_matrix_seiz{1,i}','--')
%     legend('Clean', 'Seiz')
    
    AreaROC_clean(i) = trapz(FPR_matrix_CleanTrain{1,i},TPR_matrix_CleanTrain{1,i});
    AreaROC_seiz(i) = trapz(FPR_matrix_seiz{1,i},TPR_matrix_seiz{1,i});
end


%% 
figure()
plot(AreaROC_clean)
hold on 
plot(AreaROC_seiz)
legend('Clean', 'Seiz')
xlabel('Patient')
ylabel('Area under the curve')




% saveas(h, sprintf('ROC_PC_CleanTrain_%s', plot_name),'epsc')