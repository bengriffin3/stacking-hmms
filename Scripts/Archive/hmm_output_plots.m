%% Interesting things to plot
% Functional connectivity (i.e. covariance) between 50 brain regions for each state
% (e.g. up to 8 states here)
% Load data
clear; clc;
DirResults = ['Dokumenter/MATLAB/HMMMAR_BG/HMMMAR Results/'];
DirFC = ['FC_HMM_zeromean_1_covtype_full_vary_states_3_14/'];
load([DirResults DirFC 'HMMs_r3_GROUP.mat'])
K = hmm.K;

figure(1)
for i = 1:K
    subplot(3,3,i)
    imagesc(getFuncConn(hmm,i))   
    title('Functional Connectivity, State',i)
    xlabel('Brain Region'); ylabel('Brain Region')
    colorbar
end

% mean activation for each brain region (given for each state)
figure(2) 
imagesc(meanActivations)
colorbar()
title('Mean activation for each brain region')
xlabel('State, K'); ylabel('Brain Region')

x = 1:120; % we plot the following graphs for the first 'x' time points
% e.g. 120 = first subject, first session, first 120 time points
% i.e. if scans taken every 2 seconds, that's first 240 seconds = 4 minutes

% State Time Courses (i.e. probability of each state being active at time t)
figure(3)
for i = 1:K
    plot(x,Gamma(x,i))
    hold on
end
title('Probability of each state being active at time t')
xlabel('Time, t'); ylabel('Probability')
set(gca,'ylim',[-0.2 1.2]);
legend('State 1','State 2','State 3','State 4','State 5','State 6','State 7','State 8')


% Viterbi path (most likely sequence of states)
figure(4)
subplot(2,1,1)
plot(x,vpath(x))
%scatter(x,vpath(x),'x')
title('Most probable state sequence')
set(gca,'ylim',[0 hmm.K+1])
xlabel('Time, t'); ylabel('State, K')
set(gca,'ylim',[0 hmm.K+1]);


% Most active states
subplot(2,1,2) 
[GC,GR] = groupcounts(vpath(x));
b = bar(GR,GC);
xlabel('State, K'); ylabel('Number of times active')
title('Number of times each state is active in viterbi path')
xtips2 = b.XEndPoints;
ytips2 = b.YEndPoints;
labels2 = string(round(b.YData,2));
text(xtips2,ytips2,labels2,'HorizontalAlignment','center',...
    'VerticalAlignment','bottom')
set(gca,'ylim',[0 max(GC)+5]);


% How free energy reduced per iteration of (HMM?)
figure(5)
plot(1:length(fehist),fehist)
title('Free energy reduction over iterations of algorithm')
xlabel('Iteration'); ylabel('Free History')


% State transition probability matrices (ignoring diagonal entries). In
% reality, the diagonal elements are the largest by a long way because
% states genereally transition to themselves (which is why we ignore them,
% because it distorts the rest of the matrix).
figure(6)
P_plot = hmm.P;
%P_plot = P_plot - diag(diag(P_plot));
imagesc(P_plot - diag(diag(P_plot)))
xlabel('State, K'); ylabel('State, K')
title('State Transition Probability Matrix')
colorbar()


% Probability of starting in each state (Initial State Probabilities)
figure(7)
b = bar(hmm.Pi);
xlabel('State, K'); ylabel('Probability')
title('Initial State Probabilities')
xtips2 = b.XEndPoints;
ytips2 = b.YEndPoints;
labels2 = string(round(b.YData,2));
text(xtips2,ytips2,labels2,'HorizontalAlignment','center',...
    'VerticalAlignment','bottom')


% Mean fractional occupancy per state (i.e. what % of the time are we
% in each state)
figure(8)
b = bar(mean(FOgroup)*100);
title('Fractional Occupancy (Group level)')
xlabel('State, K'); ylabel('%');
ylim([0 round(max(mean(FOgroup)*100))+2]);
xtips2 = b.XEndPoints;
ytips2 = b.YEndPoints;
labels2 = string(round(b.YData,1));
text(xtips2,ytips2,labels2,'HorizontalAlignment','center',...
    'VerticalAlignment','bottom')


% Max fractional occupancy per subject
% i.e. maximum length of time a subject spent in one state. If this is too
% high, then it is an indication that the HMM is doing a poor job in
% finding the data dynamics (e.g. if maxFO = 1, then states are being assigned to describe entire subjects).)
figure(9)
subplot(2,1,1)
bar(maxFO);
title('Maximum Fractional Occupancy')
xlabel('Subject'); ylabel('Proportion of time')
% The state switching rate can be viewed as a measure of stability per
% subject
subplot(2,1,2)
bar(switchingRate);
title('Switching Rate')
xlabel('Subject'); ylabel('Switching Rate')

% Below I have plotted the mean activation for each state and functional 
% connectivity between states by looking into the structure hmm.
% I have already plotted these above, and they are very similar but not 
% exactly the same - not sure why? The above one is taken from the 
% meanActivations variable that was determined in the code after running 
% the HMM, whereas this is from the means stored in the hmm structure.
% figure
% w_vec = zeros(50,8)
% for i = 1:K
%     w = hmm.state(i).W
%     w_store(:,i) = w.Mu_W
% end
% figure
% for i = 1:K
%     subplot(3,3,i)
%    w = hmm.state(i).W;
%    imagesc(w.S_W)
%    colorbar()
% end
