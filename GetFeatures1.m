function E = GetFeatures1(Data,length_window)

    % Input:
    %       Data: EEG data

    % Output: 
    %       E: The energy matrix

    % Create windows
    [W, num_window] = Frames(Data, length_window);

    % Transform the training and test windows and calculate the energy e
    num_var = size(Data,1);
    E = zeros(num_var,num_window);

    for i = 1:num_window
        X = W(:,:,i);
        e = sum(X'.^2);
        E(:,i) = e';
    end 

end 

