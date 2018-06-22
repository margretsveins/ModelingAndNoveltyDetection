train = Y_train_F1{1,10}(1:6,:);
test = Y_test_F1{1,10}(1:6,:);

train_mean = mean(train,2); 
train_std = std(train')';
train = (train-train_mean)./train_std;
test = (test-train_mean)./train_std;
% subplot(2,1,2)
% plot(test(1,:),test(2,:),'.')   
% hold on 
% plot(train(1,:), train(2,:), 'r.')
data = [train, test];

% [K_vec, y, sig2]= model2_new(data, train,1);
[K_vec, y, sig2, Mahalanobis_dist_vec]= model2_new_cooling(data, train,1);

%  model2(data, train, 2, 10, 1)
figure()
plot(K_vec)

figure()
subplot(3,1,1)
plot(sig2)
hold on 
plot(change)
subplot(3,1,2)
plot(sig2_diff)
subplot(3,1,3) 
for i = 1:length(sig2)
    plot(abs(sig2_cell{1,i}(1:end-1)-sig2_cell{1,i}(2:end)))
    change(i) = sum(abs(sig2_cell{1,i}(1:end-1)-sig2_cell{1,i}(2:end)))
    hold on 
end

figure()
plot(Mahalanobis_dist_vec)
hold on 
plot(Q_t_vec)

clear Mahalanobis_dist Outlier min_mahala
Outlier = [];
Threshold = 2*log(1./0.4);
for i = 1:length(test)
    for q = 1:K_vec(end)
        Mahalanobis_dist(q) = (test(i)-y(q,:))*(1/sig2(q))*(test(i)-y(q,:))';
    end 

    if min(Mahalanobis_dist) >= Threshold
        Outlier = [Outlier i];
    end
    
    min_mahala(i) =  min(Mahalanobis_dist);
end
length(Outlier)
figure()
plot(min_mahala)
hold on 
plot(repmat(Threshold, 1,length(min_mahala)))
xlabel('t')
ylabel('min mahala')

figure()
labels = {'e1', 'e2','e3', 'e4','e5', 'e6'};
num_plot = 6;
counter = 1;
for i = 1:num_plot
    for j = 1:num_plot
        subplot(num_plot,num_plot,counter)
        plot(test(i,:),test(j,:),'b.'),hold on,
        plot(train(i,:),train(j,:),'r.'),hold on,
        for q = 1:length(Outlier)
            plot(test(i,Outlier(q)), test(j,Outlier(q)), 'm+')
        end 
        for k=1:K_vec(end),
           plot(y(k,i),y(k,j),'g*'),
           plot(y(k,i)+sqrt(sig2(k))*sin(2*pi*(0:31)/30),   y(k,j)+sqrt(sig2(k))*cos(2*pi*(0:31)/30),'g')
        end        
        xlabel(labels{i})
        ylabel(labels{j})
        counter = counter + 1;        
    end
end

%% 
Outlier = [];
Threshold = 2*log(1./0.01);
for i = 1:length(test)
    for q = 1:K_vec(end)
        Mahalanobis_dist(q) = (test(i)-y(q,:))*(1/sig2(q))*(test(i)-y(q,:))';
    end 

    if min(Mahalanobis_dist) >= Threshold
        Outlier = [Outlier i];
    end
end

figure()
plot(test(1,:),test(2,:),'.')   
hold on
for q = 1:length(Outlier)
    plot(test(1,Outlier(q)), test(2,Outlier(q)), 'm+')
end 
plot(train(1,:), train(2,:), 'r.')
hold on 
plot(y(:,1),y(:,2),'g*')
hold on 
for k=1:K_vec(end),
plot(y(k,1),y(k,2),'g*'),
plot(y(k,1)+sqrt(sig2(k))*sin(2*pi*(0:31)/30),   y(k,2)+sqrt(sig2(k))*cos(2*pi*(0:31)/30),'g')
end  


Outlier = [];
Threshold = 2*log(1./0.01);
for i = 1:length(test)
    for q = 1:K_vec(end)
        Mahalanobis_dist(q) = (test(i)-y(q,:))*(1/sig2(q))*(test(i)-y(q,:))';
    end 

    if min(Mahalanobis_dist) >= Threshold
        Outlier = [Outlier i];
    end
end

out_plot = zeros(1,length(test));
out_plot(Outlier) = 1;
figure()
plot(out_plot, '*')
sum(out_plot)

%% 

ten_worst = zeros(1,length(test));
ten_worst(Outlier) = 1; 
Seizure_time = seizure{1,10}

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


%%
e_max = ones(1,3600)*0.2;

e_growth = (1:3600)/3600;

e_t = [e_max; e_growth];

e_t_min = min(e_t);

figure()
subplot(2,1,1)
plot(e_t_min)
subplot(2,1,2)
plot(2*log(1./e_t_min))

%% 
figure()
plot(E_train)
hold on 
plot(E_test)
hold on 
plot(K_vec)
legend('Train', 'Test', 'K')

%% 
figure()
for i = 1:length(Y_test_F1)
    subplot(2,5,i)
    test = Y_test_F1{1,i}(1:2,:);
    train = Y_train_F1{1,i}(1:2,:);
    plot(test(1,:),test(2,:),'.')   
    hold on 
    plot(train(1,:), train(2,:), 'r.')    
    title(['F1 - ' title_plot{1,i}])
    legend('Test','Train')
    xlabel('e1')
    ylabel('e2')
    
end 

figure()
for i = 1:length(Y_test_F1)
    subplot(2,5,i)
    test = Y_test_F2{1,i}(1:2,:);
    train = Y_train_F2{1,i}(1:2,:);
    plot(test(1,:),test(2,:),'.')   
    hold on 
    plot(train(1,:), train(2,:), 'r.')
    title(['F1 - ' title_plot{1,i}])
    legend('Test','Train')
    xlabel('e1')
    ylabel('e2')
end 

%% 
e_max = ones(1,3600)*0.2;
e_growth = (1:3600)/3600;
e_t = [e_max; e_growth.*e_max];
e_t_min = min(e_t);

Q_t = 2*log(1./e_t_min);

figure()
subplot(2,1,1)
plot(e_t_min)
subplot(2,1,2)
plot(Q_t)
