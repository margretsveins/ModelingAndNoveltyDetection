function log_plot(log_p_test, threshold, Plot_title, Seizure_time)
    bin_size = 10;
    [I,J] = find(threshold > log_p_test);
      

    ten_worst = zeros(1, length(log_p_test));
    ten_worst(log_p_test <= threshold) = 1;
    ten_worst(log_p_test > threshold) = 0;
    
    log_worst = zeros(1,length(log_p_test));
    log_worst(log_p_test <= threshold) = log_p_test(log_p_test <= threshold); 
    
 
    h = figure()
    subplot(2,1,1)
        b1 = bar(ten_worst, 'b','EdgeColor', 'b', 'EdgeAlpha', 0.05)       
        hold on
        bar(Seizure_time, ten_worst(Seizure_time), 'r','EdgeColor', 'r')        
        legend('Non-Seizure','Seizure')
        alpha(b1, 0.5)    
        b1.EdgeAlpha = 0.10
    hold on 
    title(['Most abnormal datapoints(' num2str(threshold) '%), or outliers, for ' Plot_title])
    axis([0 length(ten_worst) 0 1.2])
    xlabel('Time (8s)')
    yticks([1])
    yticklabels('Outliers')
    subplot(2,1,2)
     histogram(J,floor(length(ten_worst)/bin_size), 'FaceColor', 'k', 'EdgeColor', 'k')
    axis([0 length(ten_worst) 0 bin_size])
    title(['Outliers histogram, where each bin size is ' num2str(bin_size) 'seconds'])  
    xlabel('Time (s)')
    ylabel('Numbers of outliers')  
    
%       saveas(h, sprintf('ZeroOneSpectral_CleanTrain_%s', plot_name),'epsc')
end 