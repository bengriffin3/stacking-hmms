% Initialising the parameters:
addpath(genpath('.')) % assuming we are in the HMM-MAR directory
K = 4; % number of states
ndim = 3; % number of channels
N = 10; % number of trials
Fs = 200;
T = 10000 * ones(N,1); % number of data points

% Creating a HMM-MAR structure, with Gaussian observations (order equal to 0):
hmmtrue = struct();
hmmtrue.K = K;
hmmtrue.state = struct();
hmmtrue.train.covtype = 'full';
hmmtrue.train.zeromean = 0;
hmmtrue.train.order = 0;
hmmtrue.train.lowrank = 0;

r0 = randn(ndim); % common factor
for k = 1:K
    hmmtrue.state(k).W.Mu_W = rand(1,ndim);
    r = randn(ndim);
    hmmtrue.state(k).Omega.Gam_rate = 1000 * (0.75 * r0' * r0 + 0.25 * r' * r + eye(ndim));
    hmmtrue.state(k).Omega.Gam_shape = 1000;
end

hmmtrue.P = rand(K) + 100 * eye(K);  
for j=1:K,
    hmmtrue.P(j,:) = hmmtrue.P(j,:) ./ sum(hmmtrue.P(j,:));
end;
hmmtrue.Pi = ones(1,K); %rand(1,K);
hmmtrue.Pi = hmmtrue.Pi./sum(hmmtrue.Pi);

% Generating some data:
[X,T,Gammatrue] = simhmmmar(T,hmmtrue,[]);

% Training a HMM model with Gaussian states:
options = struct();
options.K = K; 
options.Fs = Fs; 
options.covtype = 'full';
options.order = 0;
options.DirichletDiag = 2; 
options.zeromean = 0;
options.verbose = 1;

[hmm, Gamma, Xi, vpath] = hmmmar(X,T,options);

% Plot (a segment of the) true state path
figure; subplot(3,1,1)
plot(Gammatrue(1:1000,:)), set(gca,'Title',text('String','True state path'))
set(gca,'ylim',[-0.2 1.2]); ylabel('state #')

% Plot estimated probabilistic state time courses (note that the states' order within the HMM struct is random, so which colour is which is also random in the figure)
subplot(3,1,2)
plot(Gamma(1:1000,:)), set(gca,'Title',text('String','True state path'))
set(gca,'ylim',[-0.2 1.2]); ylabel('state #')

% Plot Viterbi path
subplot(3,1,3)
plot(vpath(1:1000)), set(gca,'Title',text('String','True state path'))
set(gca,'ylim',[0 hmm.K+1]); ylabel('state #')

% Plot ground truth covariance matrices
figure
subplot(2,4,1), imagesc(getFuncConn(hmmtrue,1)), colormap('gray'), set(gca,'Title',text('String','Simulated covariance'))
subplot(2,4,2), imagesc(getFuncConn(hmmtrue,2)), colormap('gray'), set(gca,'Title',text('String','Simulated covariance'))
subplot(2,4,3), imagesc(getFuncConn(hmmtrue,3)), colormap('gray'), set(gca,'Title',text('String','Simulated covariance'))
subplot(2,4,4), imagesc(getFuncConn(hmmtrue,4)), colormap('gray'), set(gca,'Title',text('String','Simulated covariance'))

% Plot inferred covariance matrices (not that the order of the states is arbitrary)
subplot(2,4,5), imagesc(getFuncConn(hmm,1)), colormap('gray'), set(gca,'Title',text('String','Inferred covariance'))
subplot(2,4,6), imagesc(getFuncConn(hmm,2)), colormap('gray'), set(gca,'Title',text('String','Inferred covariance'))
subplot(2,4,7), imagesc(getFuncConn(hmm,3)), colormap('gray'), set(gca,'Title',text('String','Inferred covariance'))
subplot(2,4,8), imagesc(getFuncConn(hmm,4)), colormap('gray'), set(gca,'Title',text('String','Inferred covariance'))

%%
% Training an HMM-MAR model (even when the data was generated with a HMM-Gaussian)
options = struct();
options.K = K; 
options.Fs = Fs; 
options.covtype = 'diag';
options.order = 1;
options.DirichletDiag = 2; 
options.zeromean = 1;
options.verbose = 1;

[hmm,Gamma,Xi] = hmmmar(X,T,options);

%Using cross-validation to assess this configuration of parameters:
options.cvfolds = 2;
[mcv,cv] = cvhmmmar(X,T,options);

%Re-compute the free energy:
fe = hmmfe(X,T,hmm,Gamma,Xi);
sum(fe)

%Getting the parametric MAR spectra:
options.Fs = Fs; 
options.completelags = 1;
options.MLestimation = 1; 
options.order = 20; % increase the order
options.p = 0.01;
spectral_info = hmmspectramar(X,T,[],Gamma,options);

%Plotting the MAR spectra:
plot_hmmspectra (spectral_info);

%Getting the non-parametric spectra:
options_mt = struct('Fs',Fs);
options_mt.fpass = [1 48];
options_mt.tapers = [4 7];
options_mt.p = 0;
options_mt.win = 500;
options_mt.order = 2;
spectral_info = hmmspectramt(X,T,Gamma,options_mt);

%Plotting the non-parametric spectra:
plot_hmmspectra (spectral_info);

%Getting different measures about the state dynamics:
FO = getFractionalOccupancy (Gamma,T,hmm.train); % state fractional occupancies per session
LifeTimes = getStateLifeTimes (Gamma,T,hmm.train); % state life times
Intervals = getStateIntervalTimes (Gamma,T,hmm.train); % interval times between state visits
SwitchingRate =  getSwitchingRate(Gamma,T,hmm.train); % rate of switching between stats