function E = getFeatures_SBS2(eeg_data, nwin, sample_freq, length_window)
        
%E = zeros(1,84);
        [pxx,f] = pwelch(eeg_data, nwin, [], [], sample_freq);
        % freq [0,3[
        E(1:14) = sum(pxx(f<3,:));
        % freq [3,5[
        E(15:28) = sum(pxx( (f>=3) == (f < 5),:));
        % freq [5,10[ 
        E(29:42) = sum(pxx( (f>=5) == (f < 10),:));
        % freq [10,21[
        E(43:56) = sum(pxx( (f>=10) == (f < 21),:));
        % freq [21,44[
        E(57:70) = sum(pxx( (f>=21) == (f < 44),:));
        % Energy 
        E(71:84) = sum(eeg_data.^2)/(length_window);
        
%         E(1,1:21,i) = sum(pxx(f<3,:));
%         % freq [3,5[
%         E(2,22:42,i) = sum(pxx( (f>=3) == (f < 5),:));
%         % freq [5,10[
%         E(3,43:63,i) = sum(pxx( (f>=5) == (f < 10),:));
%         % freq [10,21[
%         E(4,64:84,i) = sum(pxx( (f>=10) == (f < 21),:));
%         % freq [21,44[
%         E(5,85:105,i) = sum(pxx( (f>=21) == (f < 44),:));
%         j = j+21;

end 
