clear; clc
Quantile_Norm = 0;
vars = dlmread(['vars.txt'] ,' ');
load headers_with_category.mat

p = size(vars,2);
twins = dlmread(['twins.txt'],' ');
twins = twins(2:end,2:end);
grotKEEP = true(size(vars,1),1);

% Remove two subjects
grotKEEP(find(vars(:,1)==376247))=0; % remove subjects that PALM can't cope with
grotKEEP(find(vars(:,1)==168240))=0; % remove subjects that PALM can't cope with
% PALM = Permutation Analysis of Linear Models (see FSL Wiki)

% 18/10 update, test for 50 subjects so set all remaining ones to 0
grotKEEP(21:end) = 0;


twins = twins(grotKEEP,grotKEEP);
confounds = [3 8]; % sex, motion
conf = zscore(vars(grotKEEP,confounds));
% vars = vars(grotKEEP,:);

if Quantile_Norm
    conf = quantileNormalisation(conf);
end

% Now we remove variables that 
keep = true(1,p);
for BV = 1:p
    Y=vars(:,BV);
    if sum(~isnan(Y))<500, keep(BV) = false; continue; end
end



%%
% 
type_beh = [];
Y_all = [];
% 
% % 1. Demographical
% to_use = false(p,1);
% to_use(strcmp(headers(:,2),'Demographics')) = true;
% to_use(1) = false; to_use(2) = false; % remove ID and recon
% to_use(3) = false; to_use(8) = false; % remove sex and motion
% to_use(~keep) = false;
% Y = vars(grotKEEP,to_use);
% if Quantile_Norm
%     Y = quantileNormalisation (Y);
% end
% Y_all = [Y_all Y];
% type_beh = [type_beh ones(1,size(Y,2))];

% 2. Intelligence
to_use = false(p,1);
to_use(strcmp(headers(:,2),'Fluid Intelligence')) = true;
to_use(strcmp(headers(:,2),'Language/Reading')) = true;
to_use(strcmp(headers(:,2),'Language/Vocabulary')) = true;
to_use(strcmp(headers(:,2),'Processing Speed')) = true;
to_use(strcmp(headers(:,2),'Spatial Orientation')) = true;
to_use(strcmp(headers(:,2),'Sustained Attention')) = true;
to_use(strcmp(headers(:,2),'Verbal Episodic Memory')) = true;
to_use(strcmp(headers(:,2),'Working Memory')) = true;
to_use(strcmp(headers(:,2),'Episodic Memory')) = true;
to_use(strcmp(headers(:,2),'Executive Function/Cognitive Flexibility')) = true;
to_use(strcmp(headers(:,2),'Executive Function/Inhibition')) = true;
to_use(strcmp(headers(:,2),'Alertness')) = true;
to_use(510) = true; % Language_Task_Acc
to_use(518) = true; % Relational_Task_Acc
to_use(545) = true; % Working memory task acc
to_use(~keep) = false;
Y = vars(grotKEEP,to_use);
if Quantile_Norm
    Y = quantileNormalisation (Y);
end
Y_all = [Y_all Y];
type_beh = [type_beh 2*ones(1,size(Y,2))];
% % 
% % % 5. Affective variables
% to_use = false(p,1);
% to_use(strcmp(headers(:,2),'Negative Affect')) = true;
% to_use(strcmp(headers(:,2),'Psychological Well-being')) = true;
% to_use(strcmp(headers(:,2),'Social Relationships')) = true;
% to_use(strcmp(headers(:,2),'Stress and Self-efficacy')) = true;
% to_use(~keep) = false;
% Y = vars(grotKEEP,to_use);
% if Quantile_Norm
%     Y = quantileNormalisation (Y);
% end
% Y_all = [Y_all Y];
% type_beh = [type_beh 3*ones(1,size(Y,2))];
% % 
% % 6. Personality
% to_use = false(p,1);
% to_use(strcmp(headers(:,2),'Personality')) = true;
% to_use(~keep) = false;
% Y = vars(grotKEEP,to_use);
% if Quantile_Norm
%     Y = quantileNormalisation (Y);
% end
% Y_all = [Y_all Y];
% type_beh = [type_beh 4*ones(1,size(Y,2))];
% 
% % % 7. Anatomy
% % % to_use = false(p,1);
% % % to_use(strcmp(headers(:,2),'FreeSurfer')) = true;
% % % to_use(~keep) = false;
% % % Y = vars(grotKEEP,to_use);
% % % if Quantile_Norm
% % %     Y = quantileNormalisation (Y);
% % % end
% % % Y_all = [Y_all Y];
% % % type_beh = [type_beh 5*ones(1,size(Y,2))];
% % % 
% % 8. Sleep
% to_use = false(p,1);
% to_use(strcmp(headers(:,2),'Sleep')) = true;
% to_use(~keep) = false;
% Y = vars(grotKEEP,to_use);
% if Quantile_Norm
%     Y = quantileNormalisation (Y);
% end
% to_use = find(to_use);
% Y_all = [Y_all Y];
% type_beh = [type_beh 6*ones(1,size(Y,2))];

Y = Y_all;

[GC,GR] = groupcounts(type_beh');

vars = Y;

%% Make design matrix
% 
% outdir = '/home/diegov/MATLAB/experiments_HCP1200';
% 
% for r = 1:5
%     
%     dat = load([outdir '/hmms/ICA50/K8HMMs_r' num2str(r) '_FULLGAUSS.mat']);
%     
%     for j = 1:length(HMMs_dualregr)
%         [x,ii] = vectorize_HMM(dat.HMMs_dualregr{j},dat.FOdual(j,:));
%         if j == 1
%             X = zeros(length(HMMs_dualregr),length(x));
%             I = zeros(length(HMMs_dualregr),length(x),'single');
%         end
%         X(j,:) = x; I(j,:) = ii;
%     end
%     for i = 1:size(X,2)
%         ind = I(:,i) == -1;
%         X(ind,i) = median(X(~ind,i));
%     end
%     
%     X = X(grotKEEP,:); I = I(grotKEEP,:);
%     
%     save([outdir '/regression/K8HMMs_r' num2str(r) '_FULLGAUSS.mat'],...
%         'X','I','Y','type_beh',' ','conf');
%     
%     r
%     
% end
% 
% 
% %% Distances for kernel regression
% 
% outdir = '/home/diegov/MATLAB/experiments_HCP1200';
% 
% for r = 1:5
%     D_KL = zeros(N);
%     out = load([outdir '/hmms/ICA50/K8HMMs_r' num2str(r) '_FULLGAUSS.mat']);
%     for n1 = 1:N-1
%         for n2 = n1+1:N
%             % FO is contained in TPC; TPC is contained in HMM
%             D_KL(n1,n2) = (hmm_kl(out.HMMs_dualregr{n1},out.HMMs_dualregr{n2}) ...
%                 + hmm_kl(out.HMMs_dualregr{n2},out.HMMs_dualregr{n1}))/2;
%             D_KL(n2,n1) = D_KL(n1,n2);
%         end
%     end
%     D_KL = D_KL(grotKEEP,grotKEEP);
%     save([outdir '/regression/K8HMMs_r' num2str(r) '_FULLGAUSS.mat'],'D_KL','-append');
%     r
% end


% %% Group the beahvioural categories
% load('headers_with_category');
% headers_grouped_category = cell(679,4);
% headers_grouped_category(:,1:2) = headers;
% 
% 
% x = headers_grouped_category(:,3);
% x(strcmp(headers(:,2),'Demographics') == 1) = {'Demographics'};
% x(strcmp(headers(:,2),'Negative Affect') == 1) = {'Affective'};
% x(strcmp(headers(:,2),'Psychological Well-being') == 1) = {'Affective'};
% x(strcmp(headers(:,2),'Social Relationships') == 1) = {'Affective'};
% x(strcmp(headers(:,2),'Stress and Self-efficacy') == 1) = {'Affective'};
% x(strcmp(headers(:,2),'Personality') == 1) = {'Personality'};
% x(strcmp(headers(:,2),'FreeSurfer') == 1) = {'Anatomy'};
% x(strcmp(headers(:,2),'Sleep') == 1) = {'Sleep'};
% x(strcmp(headers(:,2),'Fluid Intelligence') == 1) = {'Intelligence'};
% x(strcmp(headers(:,2),'Language/Reading') == 1) = {'Intelligence'};
% x(strcmp(headers(:,2),'Language/Vocabulary') == 1) = {'Intelligence'};
% x(strcmp(headers(:,2),'Processing Speed') == 1) = {'Intelligence'};
% x(strcmp(headers(:,2),'Spatial Orientation') == 1) = {'Intelligence'};
% x(strcmp(headers(:,2),'Sustained Attention') == 1) = {'Intelligence'};
% x(strcmp(headers(:,2),'Verbal Episodic Memory') == 1) = {'Intelligence'};
% x(strcmp(headers(:,2),'Working Memory') == 1) = {'Intelligence'};
% x(strcmp(headers(:,2),'Episodic Memory') == 1) = {'Intelligence'};
% x(strcmp(headers(:,2),'Executive Function/Cognitive Flexibility') == 1) = {'Intelligence'};
% x(strcmp(headers(:,2),'Executive Function/Inhibition') == 1) = {'Intelligence'};
% x(strcmp(headers(:,2),'Alertness') == 1) = {'Intelligence'};
% x(strcmp(headers(:,2),'Language Task') == 1) = {'Intelligence'}; % to_use(510) = true; % Language_Task_Acc
% x(strcmp(headers(:,2),'Relational Task') == 1) = {'Intelligence'}; % to_use(518) = true; % Relational_Task_Acc
% x(strcmp(headers(:,2),'Working Memory Task') == 1) = {'Intelligence'}; % to_use(545) = true; % Working memory task acc
% 
% x(strcmp(headers(:,2),'Alcohol Use') == 1) = {'Other'};
% x(strcmp(headers(:,2),'Alcohol Use and Dependence') == 1) = {'Other'};
% x(strcmp(headers(:,2),'Confound') == 1) = {'Other'};
% x(strcmp(headers(:,2),'Emotion Recognition') == 1) = {'Other'};
% x(strcmp(headers(:,2),'Emotion Task') == 1) = {'Other'};
% x(strcmp(headers(:,2),'Family History of Psychiatric and Neurologic Disorders') == 1) = {'Other'};
% x(strcmp(headers(:,2),'Gambling Task') == 1) = {'Other'};
% x(strcmp(headers(:,2),'Illicit Drug Use') == 1) = {'Other'};
% x(strcmp(headers(:,2),'Marijuana Use and Dependence') == 1) = {'Other'};
% x(strcmp(headers(:,2),'Motor') == 1) = {'Other'};
% x(strcmp(headers(:,2),'Physical Health') == 1) = {'Other'};
% x(strcmp(headers(:,2),'Psychiatric History') == 1) = {'Other'};
% x(strcmp(headers(:,2),'Psychiatric and Life Function') == 1) = {'Other'};
% x(strcmp(headers(:,2),'Self-regulation/Impulsivity') == 1) = {'Other'};
% x(strcmp(headers(:,2),'Sensory') == 1) = {'Other'};
% x(strcmp(headers(:,2),'Social Task') == 1) = {'Other'};
% x(strcmp(headers(:,2),'Substance Abuse') == 1) = {'Other'};
% x(strcmp(headers(:,2),'Tobacco Use') == 1) = {'Other'};
% x(strcmp(headers(:,2),'Tobacco Use and Dependence') == 1) = {'Other'};
% x(strcmp(headers(:,2),'Vision') == 1) = {'Other'};
% headers_grouped_category(:,3) = x;
% 
% %% Numbering Categories
% % Demographics = 1; Intelligence = 2; Affective = 3; Personality = 4
% % Sleep = 5; Anatomy = 6; Other = 7
% y = headers_grouped_category(:,4);
% y(strcmp(headers_grouped_category(:,3),'Demographics') == 1) = {1};
% y(strcmp(headers_grouped_category(:,3),'Intelligence') == 1) = {2};
% y(strcmp(headers_grouped_category(:,3),'Affective') == 1) = {3};
% y(strcmp(headers_grouped_category(:,3),'Personality') == 1) = {4};
% y(strcmp(headers_grouped_category(:,3),'Sleep') == 1) = {5};
% y(strcmp(headers_grouped_category(:,3),'Anatomy') == 1) = {6};
% y(strcmp(headers_grouped_category(:,3),'Other') == 1) = {7};
% headers_grouped_category(:,4) = y;
% 
% category_index_vec = cell2mat(headers_grouped_category(:,4));
% x = category_index_vec;
% % create vector of 1s where intelligence variables are
% % x(x~=2) = 0;
% % x(x == 2) = 1;
% % intelligence_variable_vec = find(x==1)
% 
% %%
% % let's remove variables where we have more than 100 subjects with NaN recorded
% nan_variables = find(sum(isnan(vars))' > 100);
% vars(:,nan_variables) = []; % these variables are so dodgy let's just ignore them
% x(nan_variables) = [];
% [B,idx] = sort(x);
% 


