function [y,sig2,p]=gm_init(x,K,method)
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
if method==1,
  [a,index]=sort(rand(Ntrain,1));
  y=x(index(1:K),:);
  sig2=5*trace(zz)*rand(K,1)/D;    %variances
end
% method (2) draw random points from normal dist
if method==2,
  y=ones(K,1)*mean(x);
  y=y+randn(K,D)*sqrtm(zz); 
  sig2=10*trace(zz)*rand(K,1)/D;    %variances
end
% method (3) select random seed points among training points narrow variance
if method==3,
   [a,index]=sort(rand(Ntrain,1));
   y=x(index(1:K),:);
   sig2=0.004*trace(zz)*rand(K,1)/D;    %variances
end
% 
%

