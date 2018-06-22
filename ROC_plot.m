function [FPR, TPR] = ROC_plot(E_train, E, Frac, seizure)

% True posiitive rate (TPR): True positive / Condition positive
% False positive rate (FPR): False positive / Conditoin negative

TPR = zeros(1,length(Frac));
FPR = zeros(1,length(Frac));
for i = 1:length(Frac)
    [~, ~, ~, is_outlier] = log_normal(E_train, E, 'Plot', false, 'Patient 20: Seizure', Frac(i));
    True_posative = sum(is_outlier(seizure));
    Condition_positive = length(seizure);
    False_positive = sum(is_outlier) - True_posative;
    Condition_negative = length(is_outlier)-Condition_positive;

    TPR(i) = True_posative/Condition_positive;
    FPR(i) = False_positive/Condition_negative;
end 

end 