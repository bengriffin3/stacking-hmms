%% METAFEATURE STORAGE SCRIPT
% load data
clc
datadir = 'C:\Users\au699373\OneDrive - Aarhus Universitet\Dokumenter\MATLAB\HMMMAR_BG\Behavioural Variables/';
vars = dlmread([ datadir 'vars.txt'] ,' ');
grotKEEP(find(vars(:,1)==376247))=0; % remove subjects that PALM can't cope with
grotKEEP(find(vars(:,1)==168240))=0; % remove subjects that PALM can't cope with
grotKEEP(find(vars(:,1)==122418))=0; % remove subjects that PALM can't cope with
build_regression_data_V3_SC;
load hcp1003_RESTall_LR_groupICA50.mat
f = data(grotKEEP);
n_sessions= 4;
n_timepoints = 1200;
T_subject{1} = repmat(n_timepoints,n_sessions,1)';
T = repmat(T_subject,size(vars,1),1);


%%

% METADATA AT GROUP LEVEL
% Diego suggested: maximum fractional occupancy, free energy, number of
% states
% Let's begin by combining the metafeature data together across repetitions
% of the HMM
K_vec = repmat(12,15,1);
%K_vec = repmat(repelem([3 7 11 15],1,4),1,4)';
%DirOut = 'C:\Users\au699373\OneDrive - Aarhus Universitet\Dokumenter\MATLAB\HMMMAR_BG\HMMMAR Results\Changing_models\zeromean_1_covtype_full_vary_states_3_reps\';
DirOut = 'C:\Users\au699373\OneDrive - Aarhus Universitet\Dokumenter\MATLAB\HMMMAR_BG\HMMMAR Results\FEB_2022\Same_states\';
repetitions = length(K_vec);
maxFO_all_reps = zeros(1001,repetitions);
fehist_all_reps = zeros(100,repetitions);
%K_all_reps = zeros(1001,repetitions);
K_v_path = zeros(1001,repetitions);
switchingRate_all_reps = zeros(1001,repetitions);
vpath_all_reps = cell(1001,repetitions);
Gamma_all_reps = cell(1001,repetitions);


for i = 1:repetitions
    i
    K = K_vec(i);
    %load ([DirOut 'HMMs_r' num2str(i) '_GROUP_states_' num2str(K) '.mat'])
    %load ([DirOut 'HMMs_r' num2str(i) '_d'  num2str(i) '_GROUP.mat'])
    load ([DirOut 'HMMs_r' num2str(i) '_GROUP.mat'])

%     if length(fehist) == 99
%         fehist = [ fehist fehist(end)];
%     end
    for j = 1:1001
         %vpath_all_reps{j,i} = vpath(((4800*j)-4799):4800*j);
        K_v_path(j,i) = length(unique(vpath(((4800*j)-4799):4800*j)));
        %Gamma_all_reps{j,i} = Gamma(((4800*j)-4799):4800*j,:);
        

    end
    maxFO_all_reps(:,i) = maxFO;
    %fehist_all_reps(:,i) = fehist;
    %K_all_reps(i) = hmm.K; 
    switchingRate_all_reps(:,i) = switchingRate;
end
save([DirOut 'HMMs_meta_data_group.mat'],'maxFO_all_reps','K_v_path','switchingRate_all_reps')
%save([DirOut 'HMMs_meta_data_group.mat'],'maxFO_all_reps','switchingRate_all_reps')
% 
%% METADATA AT SUBJECT LEVEL


%Gammaj_subject = cell(1001,repetitions);
FOdual_subject = NaN(1001,max(K_vec),repetitions);
Entropy_subject = NaN(1001,repetitions);
likelihood_subject = NaN(1001,repetitions);

for i = 1:repetitions
    i
    K = K_vec(i);
    load([DirOut 'HMMs_r' num2str(i)  '_states_' num2str(K) '.mat']);
    %Gammaj_subject(:,i) = Gammaj;
    FOdual_subject(:,1:K,i) = FOdual;
    
    Entropy_subject(:,i) = -sum(FOdual.*log2(FOdual),2);
    
    for j = 1:1001
        j
        FOdual_sub = FOdual(j,:);
        Entropy_subject(j,i) = -sum(FOdual_sub.*log2(FOdual_sub));
        [fe,ll] = hmmfe_single_subject(f{j},T{j},HMMs_dualregr{j},Gammaj{j});
        likelihood_subject(j,i) = -sum(ll);
    end
    

end

save([DirOut 'HMMs_meta_data_subject.mat'],'FOdual_subject','Entropy_subject','likelihood_subject')

         


%FOdual_subject = cell(5,1);
%FOdual_subject{i} = FOdual;