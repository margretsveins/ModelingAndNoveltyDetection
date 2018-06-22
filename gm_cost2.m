function [E, prob_x] =gm_cost2(x,y,sig2,prob_k);
%
%function E=gm_cost(x,y,sig2,p) 
%   computes the negative log(likelihood) pr example
% 
% input:
% output:
%
[K,D]=size(y);
[N,D]=size(x);

x2=ones(K,1)*sum((x.*x)');

dist=sum((y.*y)')'*ones(1,N) + x2 -  2*y*x';   
prob_x_k=diag(1./((2*pi*sig2).^(D/2)) )*exp(-0.5*diag(1./sig2)*dist)+10e-20;
prob_x=sum(diag(prob_k)*prob_x_k);
E=-sum(log(prob_x))/N; 