d = 7
n = 50;          % Dimension of all vectors
x = 1:n;           % Optionally: specify `x` or draw from any distribution
y = randn(n,d); % Create `d` original variables in any way
rho = [0.3, 0.2, 0.1,0.4,0.2,0.3,0.5]';%rand(d,1)%0.5*ones(d,1);
threshold = 1e-12;
%[z,singular_values] = complement(y, rho, x)

mdl = fitlm(y,x); % returns a linear regression model of the responses y, fit to the data matrix X.

e = mdl.Residuals.Raw;
[U,S,V] = svd(y);
singular_values = diag(S);
nnz_svs = singular_values > threshold;
pseudo_inverse = 1./singular_values;
pseudo_inverse(~nnz_svs) = 0; % set all very small singular values to 0
pseudo_inverse_d = zeros(n,d);
pseudo_inverse_d(1:d,1:d) = diag(pseudo_inverse); % convert to matrix (rather than list of inverted singular values

y_dual = (n-1)*U * pseudo_inverse_d * V';

sigma2 = ((1 - rho)' * cov(y_dual) * rho) / var(e);

sigma = sqrt(sigma2);
z = y_dual* rho + sigma*e;

actual_correlations = corr(y,z);
target_correlation = rho;
correlation_target_comparison = [actual_correlations target_correlation]
%cbind('Actual correlations' = cor(y, z), # find pearson correlation between y and z (z is our new vector with defined correlations to all y_i) 
%      'Target correlations' = rho)







% y = [0.70783581	1.137118211	;
% -0.61470393	0.550301675	;
% 2.06474242	1.069964435	;
% 1.55383199	0.275133082	;
% 0.11427787	0.038123055	;
% -1.13551508	-1.819707821	;
% -0.36384565	2.054863658	;
% 0.47350561	-0.178345202	;
% 3.00782151	0.054091955	;
% -0.25688296	1.068386007	;
% 0.40430523	-0.924571235	;
% -1.22632412	-1.173539373	;
% 0.20722041	0.184910061	;
% -0.18108931	-0.183057134	;
% -0.81530206	0.450862106	;
% -0.6771524	-2.363972286	;
% 0.73021671	-0.513865043	;
% -0.22862171	1.593128674	;
% -1.88724766	-0.009897101	;
% 1.31401956	1.037228993	;
% 1.09883284	-0.012064549	;
% -0.77186903	-1.1371104	;
% 1.00898161	-1.095298843	;
% -0.15738248	-1.056827666	;
% 0.31831464	0.095485665	;
% -1.04954366	-1.224711927	;
% -0.1794256	0.221652907	;
% -0.57674702	0.734641372	;
% 0.30103744	0.020195629	;
% -0.55710587	0.586507342	;
% 0.29147654	0.747507217	;
% 0.29928206	-0.723310427	;
% 0.39552407	-1.936375828	;
% 0.24019182	-0.147952244	;
% -0.51285083	0.113886421	;
% 0.83872595	-2.044483671	;
% 0.95407902	0.324881736	;
% 0.44439047	-0.470851783	;
% -0.61708908	0.300102487	;
% 0.07784074	0.327223221	;
% -1.21781748	1.455231702	;
% -0.645069	1.289626607	;
% 1.17050752	0.519134836	;
% -0.74358568	0.009949596	;
% 1.75546594	0.130672697	;
% -1.08485111	0.515831477	;
% 0.27435114	-1.790816503	;
% 1.09710945	-0.95809639	;
% -0.206856	0.518229807	;
% 0.16220878	-1.29476089	];