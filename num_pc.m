function  [E_out, E_train_out, num_prin] = num_pc(E, E_train, var_explained, var_target)

num_prin = 1;
while sum(var_explained(1:num_prin)) < var_target
    num_prin = num_prin + 1;
end 

E_out = E(1:num_prin,:);
E_train_out = E_train(1:num_prin,:);

end 