%% Create distance matrices between group level HMM models 
clc
% % between HMMs
repetitions = 4;
dirichlet_test = 4;
DistHMM_group = zeros(repetitions,dirichlet_test,dirichlet_test); % subj x subj x repetitions
hmm_all = cell(repetitions,dirichlet_test);
DirOut = 'C:\Users\au699373\OneDrive - Aarhus Universitet\Dokumenter\MATLAB\HMMMAR_BG\HMMMAR Results\FEB_2022\K_prior_grid_search_5_with_folds\Zeromean_1\'; % Test folder
r_vec = 1:16;
DD_store = NaN(1,dirichlet_test);

% load and store data
for r = 1:repetitions % select number of states
    disp(r)
    
    for d = 1:dirichlet_test
        i = r_vec(d);
        out = load([DirOut 'HMMs_r' num2str(i) '_d' num2str(d)  '_GROUP.mat']);
        hmm_all{r,d} = out.hmm;
        DD_store(1,d) = hmm_all{1,d}.train.DirichletDiag;
    end
end

% Note that we can only find KL divergence between HMMs that have the same
% number of states
for r = 1:repetitions
    for i = 1:dirichlet_test-1
        disp(['Repetition ' num2str(i) ])
        for j = i+1:dirichlet_test
            DistHMM_group(r,i,j) = (hmm_kl(hmm_all{r,i},hmm_all{r,j}) ...
                + hmm_kl(hmm_all{r,j},hmm_all{r,i}))/2;
            DistHMM_group(r,j,i) = DistHMM_group(r,i,j);
        end
    end
end

%% Plot results
figure; 
for r = 1:repetitions
    %namelist = {'DD1','DD2','DD3','DD4','DD5','DD6','DD7','DD8'};
    namelist = num2cell(DD_store);
    X = 1:dirichlet_test;
    Y = cmdscale(max(squeeze(DistHMM_group(r,:,:)),0));
    subplot(2,2,r); plot(X,Y,'.')
    text(X+0.1,Y,namelist)
    title('Map of distances between HMMs using classical multidimensional scaling')
    xlabel('HMM repetition'); ylabel('KL divergence');
    legend('Dirichlet diag value')
end
%%% RESULT
% I think something in the dirichlet diag just makes a massive difference
% in the KLdivergence between HMMs cause whatever repetition, the same
% thing happens where DD = 10, 10000 are close, then the next 4 values of
% DD are the same and the final DD value is far away from everything.

