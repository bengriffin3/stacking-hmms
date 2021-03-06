
Quantile_Norm = 0;
datadir = 'C:\Users\au699373\OneDrive - Aarhus Universitet\Dokumenter\MATLAB\HMMMAR_BG\Behavioural Variables/';
vars = dlmread([ datadir 'vars.txt'] ,' ');

p = size(vars,2);

twins = dlmread([ datadir '/twins.txt'],' ');
twins = twins(2:end,2:end);
load headers_with_category.mat
grotKEEP = true(size(vars,1),1);
grotKEEP(find(vars(:,1)==376247))=0; % remove subjects that PALM can't cope with
grotKEEP(find(vars(:,1)==168240))=0; % remove subjects that PALM can't cope with
grotKEEP(find(vars(:,1)==122418))=0; % remove subjects that PALM can't cope with

% let's test 20 subjects
grotKEEP(51:end) = 0;

twins = twins(grotKEEP,grotKEEP);
confounds = [3 8]; % sex, motion
conf = zscore(vars(grotKEEP,confounds));
if Quantile_Norm
    conf = quantileNormalisation (conf);
end
keep = true(1,p);
for BV = 1:p
    Y=vars(:,BV);
    if sum(~isnan(Y))<500, keep(BV) = false; continue; end
end

type_beh = [];
Y_all = [];

%%
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

% 5. Affective variables
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
% 
% 6. Personality
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
% 7. Anatomy
% to_use = false(p,1);
% to_use(strcmp(headers(:,2),'FreeSurfer')) = true;
% to_use(~keep) = false;
% Y = vars(grotKEEP,to_use);
% if Quantile_Norm
%     Y = quantileNormalisation (Y);
% end
% Y_all = [Y_all Y];
% type_beh = [type_beh 5*ones(1,size(Y,2))];
% 
% 8. Sleep
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
vars = Y;
% 
% %% Make design matrix
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
%         'X','I','Y','type_beh','twins','conf');
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



