function [E, E_norm, PSE] = getFeatures(eeg_data, nwin, sample_freq, length_window, num_chan)
% https://dsp.stackexchange.com/questions/23689/what-is-spectral-entropy
        
        num_comp = 6;
        total_num_comp = num_chan * num_comp;
        E_norm = [];
        PSE = [];
        [pxx,f] = pwelch(eeg_data, nwin, [], [], sample_freq);
        % freq [0,3[
        E(1:6:total_num_comp) = sum(pxx(f<3,:));
        % freq [3,5[
        E(2:6:total_num_comp) = sum(pxx( (f>=3) == (f < 5),:));
        % freq [5,10[ 
        E(3:6:total_num_comp) = sum(pxx( (f>=5) == (f < 10),:));
        % freq [10,21[
        E(4:6:total_num_comp) = sum(pxx( (f>=10) == (f < 21),:));
        % freq [21,44[
        E(5:6:total_num_comp) = sum(pxx( (f>=21) == (f < 44),:));
        % Energy 
        E(6:6:total_num_comp) = sum(eeg_data.^2)/(length_window);
        
        j = 1;
        i = 1;
        while i < total_num_comp
            normalized = E(i:i+4)./sum(E(i:i+4));
            E_norm = [E_norm; normalized'];
            PSE(j,1) = -normalized * log(normalized)';
            i = i+6;  
            j = j+1;
        end 
        
        

end 
