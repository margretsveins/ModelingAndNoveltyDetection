function [y,sig2,p]=gm_init(x,K,method)

% This function is based on the funciton used in the course Non linear
% signal processing. 

% % (c) Lars Kai Hansenn

%function [y,sig2,p]=gm_init(x,K,method)
%  
%  Initialise a gaussian mixture model
%
%  input:
%
%  x: training data 
%  K: number of clusters
%  method: 1,2,3  seed points, random from covariance, seed points narrow variance
% 
%  output:
%
%  y :  K*D  matrix with initial cluster centers
%  sig2:  K*1 matrix with initial variances
%  p:    K*1 matrix with initial weights

[Ntrain,D]=size(x);

% Equal weighting of clusters
p=ones(K,1)/K;

% estimate the covariance matrix of the training set
mx=mean(x);
zz=(x-ones(Ntrain,1)*mx);
zz=zz'*zz/Ntrain;

% method (1)  select random seed points among training points
if method==1
  [a,index]=sort(rand(Ntrain,1));
  y=x(index(1:K),:);
  sig2=5*trace(zz)*rand(K,1)/D;    %variances
end
%

end 