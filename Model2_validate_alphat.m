A = Dist_test_alphat_SBS2{1,1};
dist_opt = A./sum(A(~isnan(A)));

B = num_outlier_alphat_SBS2{1,1};
num_out_opt = B./sum(B(~isnan(B)));

C = dist_opt + num_out_opt;

[M,I] = min(C(:));
[I_row, I_col] = ind2sub(size(C),I)
alpha_0 = 0.2:0.2:2;
tau_alpha = 5:0.2:7;
alpha_0_opt = alpha_0(I_col)
tau_alpha_opt = tau_alpha(I_row)

figure()
subplot(3,1,1)
imagesc(A)
colormap(gray)
colorbar
% caxis([0.45 1])
xlabel('\alpha_0', 'FontSize', 16)
ylabel('\tau_{\alpha}', 'FontSize', 16)
title('Distance from test point and the closest component')
set(gca, 'FontSize', 14, 'XTick', 1:length(alpha_0),'XTickLabel', alpha_0,'YTick', 1:length(tau_alpha),'YTickLabel', tau_alpha)

% figure()
subplot(3,1,2)
imagesc(B)
colormap(gray)
colorbar
% caxis([0 5])
xlabel('\alpha_0', 'FontSize', 16)
ylabel('\tau_{\alpha}', 'FontSize', 16)
title('Number of novelties')
set(gca, 'FontSize', 14, 'XTick', 1:length(alpha_0),'XTickLabel', alpha_0,'YTick', 1:length(tau_alpha),'YTickLabel', tau_alpha)

% figure()
subplot(3,1,3)
imagesc(C)
colormap(gray)
colorbar
% caxis([0 0.8])
xlabel('\alpha_0', 'FontSize', 16)
ylabel('\tau_{\alpha}', 'FontSize', 16)
title('Cost function')
set(gca, 'FontSize', 14, 'XTick', 1:length(alpha_0),'XTickLabel', alpha_0,'YTick', 1:length(tau_alpha),'YTickLabel', tau_alpha)



