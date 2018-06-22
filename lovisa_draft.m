
alpha_03 = y_change{1,2};
sig2_03 = sig2_change{1,2};
prob_k_03 = prob_k_change{1,2};
clear x_plot y_plot sig2_plot prob_k_plot

x_plot{1} = [];
x_plot{2} = [];
x_plot{3} = [];

y_plot{1} = [];
y_plot{2} = [];
y_plot{3} = [];

sig2_plot{1} = [];
sig2_plot{2} = [];
sig2_plot{3} = [];

prob_k_plot{1} = [];
prob_k_plot{2} = [];
prob_k_plot{3} = [];


for i = 1:length(alpha_03)
    y_t = alpha_03{1,i};
    sig2_t = sig2_03{1,i};
    prob_k_t = prob_k_03{1,i};
    for j = 1:size(y_t,1)
        if j < 4
            x_plot{j} = [x_plot{j} y_t(j,1)];
            y_plot{j} = [y_plot{j} y_t(j,2)];

            sig2_plot{j} = [sig2_plot{j} sig2_t(j,1)];
            prob_k_plot{j} = [prob_k_plot{j} prob_k_t(j,1)];
        end
    end
    
end


figure()
subplot(3,3,1)
plot(x_plot{1,1}, y_plot{1,1}, '*')
hold on 
h1 = plot(x_plot{1,1}(1), y_plot{1,1}(1), 'r*')
hold on 
h2 = plot(x_plot{1,1}(end), y_plot{1,1}(end), 'g*')
title('Center of component 1 - alpha_0 = 1.5')
xlabel('x')
ylabel('y')
legend([h1 h2], {'Location after first iteration', 'Location after last iteration'})

subplot(3,3,2)
plot(sig2_plot{1,1})
xlabel('t')
ylabel('\sigma^2_t')
title('Development of variance in component 1')

subplot(3,3,3)
plot(prob_k_plot{1,1})
xlabel('t')
ylabel('prob(k)_t')
title('Development of p(k) for component 1')

subplot(3,3,4)
plot(x_plot{1,2}, y_plot{1,2},'*')
hold on 
h1 = plot(x_plot{1,2}(1), y_plot{1,2}(1), 'r*')
hold on 
h2 = plot(x_plot{1,2}(end), y_plot{1,2}(end), 'g*')
title('Center of component 2 - alpha_0 = 1.5')
xlabel('x')
ylabel('y')
legend([h1 h2], {'Location after first iteration', 'Location after last iteration'})

subplot(3,3,5)
plot(sig2_plot{1,2})
xlabel('t')
ylabel('\sigma^2_t')
title('Development of variance in component 2')

subplot(3,3,6)
plot(prob_k_plot{1,2})
xlabel('t')
ylabel('prob(k)_t')
title('Development of p(k) for component 2')

subplot(3,3,7)
plot(x_plot{1,3}, y_plot{1,3},'*')
hold on 
h1 = plot(x_plot{1,3}(1), y_plot{1,3}(1), 'r*')
hold on 
h2 = plot(x_plot{1,3}(end), y_plot{1,3}(end), 'g*')
title('Center of component 3 - alpha_0 = 1.5')
xlabel('x')
ylabel('y')
legend([h1 h2], {'Location after first iteration', 'Location after last iteration'})

subplot(3,3,8)
plot(sig2_plot{1,3})
xlabel('t')
ylabel('\sigma^2_t')
title('Development of variance in component 3')

subplot(3,3,9)
plot(prob_k_plot{1,3})
xlabel('t')
ylabel('prob(k)_t')
title('Development of p(k) for component 3')
set(gca, 'Title', 'alpha_0 = 0.3')