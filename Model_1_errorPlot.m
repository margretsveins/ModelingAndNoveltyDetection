function Model_1_errorPlot(error_train, error_test, conf_int_train, conf_int_test,size_train, jump, title_plot, plot_name)

j = 1; 
xlabel_tick = [];
xtick = [];
while j <= length(error_test)
    xlabel_tick = [xlabel_tick size_train(j)];
    xtick = [xtick j];
    j = j + jump;
end 


h = figure()
hold on 
errorbar(error_train, 2*conf_int_train)
%plot(error_train)
hold on 
%plot(error_test)
errorbar(error_test, 2*conf_int_test)
legend('error train', 'error test')
xticks(xtick)
xticklabels(xlabel_tick)
xtickangle(45)
title(title_plot)
ylabel('|mean(log(p))|')
xlabel('Size of train set (fraction)')
% xtickformat('%g%%')
% figure()
% loglog(error_train)
% hold on
% loglog(error_test)
% legend('error train', 'error test')
% saveas(h, sprintf('LearningCurve_%s', plot_name),'epsc')


end