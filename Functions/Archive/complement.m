function [z,singular_values] = complement(y, rho, x)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

threshold = 1e-12;
d = size(y,2);
n = size(y,1);
y = y/std(y);

mdl = fitlm(y,x); % returns a linear regression model of the responses y, fit to the data matrix X.

e = mdl.Residuals.Raw;
[U,S,V] = svd(y);
singular_values = diag(S)
singular_values > threshold

z = 2;

end