% --- WITH
for k = 1:7
% --- Normal distribution
    ten_worst = ten_worst_with{1,k};


% ---- GMM
    outlier_matrix = Outlier_cell_with{1,k};
    outlier = zeros(1, length(outlier_matrix));
    outlier(sum(outlier_matrix)>5 ) = 1;
    num_outlier_with(k) = sum(outlier);

% --- Find the same 
    same = (ten_worst + outlier) == 2;
    different_with(k) = sum((ten_worst - outlier )== -1)
% ---- PLOT
    h = figure()
    subplot(2,1,1)
    b1 = bar(ten_worst, 'b','EdgeColor', 'b')     
    hold on 
    b2 = bar(same, 'r','EdgeColor', 'r')   
    title(['ML - '   title_with{1,k} ' - with IED'])
    axis([0 length(outlier) 0 1.2])
    xlabel('Time (s)')
    set(gca, 'FontSize', 14)
    subplot(2,1,2)
    b1 = bar(outlier, 'b','EdgeColor', 'b')
    hold on 
    b2 = bar(same, 'r','EdgeColor', 'r')  
    title(['GMM - ' title_with{1,k} ' - with IED'])
    axis([0 length(outlier) 0 1.2])
    xlabel('Time (s)')
    set(gca, 'FontSize', 14)

     saveas(h, sprintf('LogPlot_comp_WITH_%s', num2str(k)),'epsc')
end 


for k = 1:7
% ---- Normal dist
    ten_worst = ten_worst_non{1,k};

% ---- GMM
    outlier_matrix = Outlier_cell_non{1,k};
    outlier = zeros(1, length(outlier_matrix));
    outlier(sum(outlier_matrix)>5 ) = 1;
    num_outlier_with(k) = sum(outlier);

% --- Find the same 
    same = (ten_worst + outlier) == 2;

    different_non(k) = sum((ten_worst - outlier )== -1)

% ---- PLOT
    h = figure()
    subplot(2,1,1)
    b1 = bar(ten_worst, 'b','EdgeColor', 'b')  
    hold on
    b2 = bar(same, 'r','EdgeColor', 'r') 
    title(['ML - ' title_non{1,k} ' - non IED'])
    axis([0 length(outlier) 0 1.2])
    xlabel('Time (s)')
    set(gca, 'FontSize', 14)

    subplot(2,1,2)
    b1 = bar(outlier, 'b','EdgeColor', 'b')    
    hold on
    b2 = bar(same, 'r','EdgeColor', 'r') 
    title(['GMM - ' title_non{1,k} ' - non IED'])
    axis([0 length(outlier) 0 1.2])
    xlabel('Time (s)')
    set(gca, 'FontSize', 14)


     saveas(h, sprintf('LogPlot_comp_NON_%s', num2str(k)),'epsc')
end 
