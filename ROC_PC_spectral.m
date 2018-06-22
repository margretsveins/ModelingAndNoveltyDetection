% function [FPR_matrix, TPR_matrix, AreaROC,num_prin_vec] = ROC_PC(Data, length_window,var_target, seizure, plot_title, plot_name)
function [FPR_matrix, TPR_matrix, AreaROC,num_prin_vec, f_matrix] = ROC_PC_spectral(Data,var_explained, length_window,size_train,var_target, seizure, plot_title, plot_name)
%%  ROC curve different number of PC   
% num_window = length(Data)/length_window;
num_window = size(Data,2);
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


for i = 1:length(var_target)
    num_prin = 1;
    while sum(var_explained(1:num_prin)) < var_target(i)
        num_prin = num_prin + 1;
    end 
    Data_pc = Data(1:num_prin,:);
    Data_train_pc = Data_pc(:,1:floor(num_window*size_train));

    [FPR, TPR, f] = ROC_points(Data_train_pc, Data_pc, seizure,true);
    FPR_matrix(i,1:length(FPR)) = FPR;
    TPR_matrix(i,1:length(TPR)) = TPR;
    f_matrix(i, 1:length(f)) = f;
    AreaROC(i) = trapz(FPR,TPR);
    num_prin_vec(i) = num_prin;  
end 
if length(var_target) > 1 
    Data_train = Data(:,1:floor(num_window*size_train));
    [FPR, TPR] = ROC_points(Data_train, Data, seizure, true);
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
% 
% saveas(h, sprintf('ROC_PC_CleanTrain_F2_%s', plot_name),'epsc')


end 