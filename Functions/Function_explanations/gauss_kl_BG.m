function [D] = gauss_kl_BG (mu_q,mu_p,sigma_q,sigma_p)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   [D] = gauss_kl (mu_q,mu_p,sigma_q,sigma_p
%
%   computes the divergence 
%                /
%      D(q||p) = | q(x)*log(q(x)/p(x)) dx
%               /
%   between two k-dimensional Gaussian probability
%   densities  given means mu and Covariance Matrices sigam where the
%   Gaussian pdf is given by  
%
%              1                                     T       -1
%   p(x)= ------------------------- exp (-0.5  (x-mu)   Sigma    (x-mu)  )        %%% This is just the pdf for a multivariate Gaussian distribution with mean mu and covariance matrix sigma
%         (2*pi)^(d/2) |Sigma|^0.5
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if nargin<4
  error('Incorrect number of input arguments');
end

if length(mu_q)~=length(mu_p) % matrix of means
  error('Distributions must have equal dimensions (Means dimension)');
end
mu_q=mu_q(:);
mu_p=mu_p(:);

if size(sigma_q)~=size(sigma_p) % matrix of covariances
  error('Distributions must have equal dimensions (Covariance dimension)');
end


K=size(sigma_q,1); % number of states?
isigmap = inv(sigma_p);

D=logdet(sigma_p) - logdet(sigma_q) -K+trace(isigmap*sigma_q)+(mu_q-mu_p)'*isigmap*(mu_q-mu_p); 
D=D*0.5;
% trace is the sum of the diagonal elements
% K = number of states
% logdet(A) is just the log of the determinant of matrix A, so that logdet(sigma_p) - logdet(sigma_q) is the same as log(det(sigma_p/sigma_q))
% the derivation for the KL divergence between two multivariate Gaussian distributions can be found on p13 here: https://stanford.edu/~jduchi/projects/general_notes.pdf
% Then, we just use the formula as above

% For more info, also see: https://stats.stackexchange.com/questions/60680/kl-divergence-between-two-multivariate-gaussians