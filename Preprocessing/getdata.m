function [xtrain,xtest]=getdata(Ntrain,Ntest,noise_train, noise_test)
%function [xtrain,xtest]=getdata(Ntrain,Ntest,noise)
%
% Creates 2D data with 3 clusters of width noise
% 
% (c) Lars Kai Hansenn
%
x11=noise_train*randn(round(round(Ntrain)),1);
x21=noise_train*randn(round(round(Ntrain)),1);

x1=x11;
x2=x21;
xtrain=[x1,x2];


z11=noise_test*randn(round(Ntest),1);
z21=noise_test*randn(round(Ntest),1);

z1=z11;
z2=z21;
xtest=[z1,z2];



