  figure()
for i = 1:10
  
    subplot(2,5,i)
    imagesc(TPR_subject_cum_alpha(:,:,i))
    caxis([0 1]);
    title([title_plot{1,i} ' - Sensitivity'])
    colormap(gray)
    colorbar

   
end 
figure()
for i = 1:10
    
    subplot(2,5,i)
    imagesc(1 -FPR_subject_cum_alha(:,:,i))
    title([title_plot{1,i} ' - Specificity'])
    caxis([0 1]);
    colormap(gray)
    colorbar   
   
end 


%% 
clear C

figure()
% (t_a, a_0)
for i = 1:length(Data)
    TPR_vec = TPR_subject_cum_alpha(:,:,i);
    FPR_vec = FPR_subject_cum_alha(:,:,i);
    C = sqrt((1 - TPR_vec).^2 + FPR_vec.^2);
    [M,I] = min(C(:));
    [index_t_a, index_a_0] = ind2sub(size(C),I);
    
    TPR_opt(i) = TPR_vec(index_t_a, index_a_0);
    FPR_opt(i) = FPR_vec(index_t_a, index_a_0);
    a_0_opt(i) = alpha_0(index_a_0);
    t_a_opt(i) = t_alpha(index_t_a);
    
    subplot(2,5,i)
    imagesc(C)
    title([title_plot{1,i} ' - Ideal distance'])
    xlabel('\alpha_0', 'FontSize', 14)
    ylabel('\tau_{\alpha}','FontSize', 14)
    caxis([0 1.5]);
    colormap(gray)
    colorbar   
end 

sens_t_alpha = TPR_opt;
spec_t_alpha = (1 - FPR_opt);
a_0_opt = a_0_opt';
t_a_opt = t_a_opt';