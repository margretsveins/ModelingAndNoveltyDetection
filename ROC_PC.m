% function [FPR_matrix, TPR_matrix, AreaROC,num_prin_vec] = ROC_PC(Data, length_window,var_target, seizure, plot_title, plot_name)
function [FPR_matrix, TPR_matrix, AreaROC,num_prin_vec, f_matrix] = ROC_PC(Data, Data_train_clean,Data_train, Data_test, length_window,var_target, seizure, plot_title, plot_name)
%%  ROC curve different number of PC   
num_window = length(Data)/length_window;
% FPR_matrix = ones(length(var_target)+1,num_window+1);
% TPR_matrix = ones(length(var_target)+1,num_window+1);
if length(var_target) > 1
    FPR_matrix = ones(length(var_target)+1,num_window+1);
    TPR_matrix = ones(length(var_target)+1,num_window+1);
else 
    FPR_matrix = ones(length(var_target),num_window+1);
    TPR_matrix = ones(length(var_target),num_window+1);
end 
AreaROC = zeros(1,length(var_target));
num_prin_vec  = zeros(1,length(var_target));
size_train = 0.2;
if Data_train_clean 
    [E, E_train, var_explained] = E_matrix2(Data, Data_train, Data_test, length_window,'Plot', false ,[], 'Standardize', true);
else
    [E, E_train, var_explained] = E_matrix(Data, size_train,length_window,'Plot', false ,[], 'Standardize', true);
end 
for i = 1:length(var_target)
    [E_pc, E_train_pc, num_prin] = num_pc(E, E_train, var_explained, var_target(i));
    [FPR, TPR, f] = ROC_points(E_train_pc, E_pc, seizure,Data_train_clean);
    FPR_matrix(i,1:length(FPR)) = FPR;
    TPR_matrix(i,1:length(TPR)) = TPR;
    f_matrix(i, 1:length(f)) = f;
    AreaROC(i) = trapz(FPR,TPR);
    num_prin_vec(i) = num_prin;  
end 
if length(var_target) > 1 
    [FPR, TPR] = ROC_points(E_train, E, seizure, Data_train_clean);
    AreaROC_100 = trapz(FPR,TPR);
    AreaROC = [AreaROC AreaROC_100];
    num_prin_vec = [num_prin_vec, size(Data,1)];
    FPR_matrix(end,1:length(FPR)) = FPR;
    TPR_matrix(end,1:length(TPR)) = TPR;
end
% h = figure()
% for i = 1:length(var_target)+1
%     plot(FPR_matrix(i,:), TPR_matrix(i,:))
%     hold on
% end
% plot(FPR_matrix(i,:),FPR_matrix(i,:),'--')
% legend('80%', '85%', '90%', '95%', '99%', '100%', 'Location', 'southeast')
% title(plot_title)
% xlabel('FPR')
% ylabel('TPR')
% % 
% saveas(h, sprintf('ROC_PC_CleanTrain_%s', plot_name),'epsc')


end 