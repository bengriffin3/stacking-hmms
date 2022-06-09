function D = hmm_kl_BG (hmm_p,hmm_q)
% Computes Kullback-Leibler divergence between two Hidden Markov Model
% distributions, through an approximation (an upper bound) as proposed in
% M. Do (2003). IEEE Signal Processing Letters 10
%
% Author: Diego Vidaurre, OHBA, University of Oxford (2018)

K = length(hmm_p.Pi);  % number of states (just taken as length of initial state probabiltiy vector)
% if K~=length(hmm_q.Pi) % can only calculate this for subject described using same number of brain states
%     error(['The two HMMs must have the same number of states, ' ...
%         'and their order must correspond'])
% end
% if (hmm_p.train.order ~= hmm_q.train.order) || ... % HMM configuration must be the same between subjects too
%         (~strcmpi(hmm_p.train.covtype,hmm_q.train.covtype)) || ...
%         (length(hmm_p.train.embeddedlags) ~= length(hmm_q.train.embeddedlags)) || ...
%         (any(hmm_p.train.embeddedlags ~= hmm_q.train.embeddedlags)) || ...
%         (hmm_p.train.zeromean ~= hmm_q.train.zeromean)  
%    error('The state configuration of the two HMMs must be identical') 
% end
hmm = hmm_p; setstateoptions;
% if isfield(hmm_p.state(1),'W') % this case for Neuroimage 2021
%     ndim = size(hmm_p.state(1).W.Mu_W,2); % ndim = 0???
% else
%     ndim = size(hmm_p.state(1).Omega.Gam_rate,2);
% end
S = hmm.train.S==1; % S is a (no. channels x no. channels) matrix defining which autoregression coefficients are to be modelled and how. For Neuroimage it is a 50x50 matrix of 1s
regressed = sum(S,1)>0; % I think this is the number of channels to regress out???

D = 0; % initialise divergence
% if hmm_p.train.id_mixture
%    hmm_p.P = 1/K*ones(K); hmm_q.P = 1/K*ones(K); % P is the (K X K) state transition probability matrices
% end
nu = compute_nu (hmm_p.Pi,hmm_p.P); % weight vector 
% i.e. nu is a factor representing the weights of state k in M^1, where
% M^1 is as in
% KL(M^1||M^2) = sum_k (nu_k * KL(P{_k}{^1},P{_k}{^2} + nu_k * KL(G{_k}{^1},G{_k}{^2}
% where:
% P{_k}{^i} represents the (Dirichlet-distributed) probabilities of transitioning from state k to any other state according to model i (i.e. the k-th row of the TPM), and
% G{_k}{^i} represents the state Gaussian distribution for state k and model i

% There is no 'non-state specific stuff' in Neuroimage 2021 because we use
% a 'full' covariance matrix which has a full covariance matrix for each
% state rather than 'unique...' which has one covariance matrix for all
% states

% Non-state specific stuff
% switch train.covtype
%     case 'uniquediag' % uniquediag = one diagonal covariance matrix for all states
%         % If the covariance matrix is uniquediag, then it corresponds to a Gamma ditribution with parameters Gam_rate (1 x no. channels) and Gam_shape, which is a scalar
%         % hence why we use gamma_kl
%         for n = 1:ndim
%             if ~regressed(n), continue; end
%             D = D + gamma_kl(hmm_p.Omega.Gam_shape,hmm_q.Omega.Gam_shape, ...
%                 hmm_p.Omega.Gam_rate(n),hmm_q.Omega.Gam_rate(n));
%         end
%     case 'uniquefull'
%         % I think this isn't used in the Neuroimage 2021 paper (since uniquefull is not used in this paper) - uniquefull has one covariance matrix for all states
%         % If the covariance matrix is uniquefull, then it corresponds to a Wishart ditribution with parameters Gam_rate (no. channels x no. channels) and Gam_shape, which is a scalar
%         % hence why we use wishart_kl
%         D = D + wishart_kl(hmm_p.Omega.Gam_rate(regressed,regressed),...
%             hmm_q.Omega.Gam_rate(regressed,regressed), ...
%             hmm_p.Omega.Gam_shape,hmm_q.Omega.Gam_shape);
%     case 'pca'
%         D = D + gamma_kl(hmm_p.Omega.Gam_shape,hmm_q.Omega.Gam_shape, ...
%                 hmm_p.Omega.Gam_rate,hmm_q.Omega.Gam_rate);
% end

% State specific stuff
for k = 1:K % for each state
     % the KL divergence is the sum across all states hence why 'D = D + ...' throughout

    % Trans probabilities
    kk = hmm.train.Pstructure(k,:);
    D = D + nu(k) * dirichlet_kl(hmm_p.Dir2d_alpha(k,kk),hmm_q.prior.Dir2d_alpha(k,kk));
    % the dirichlet_kl function finds the Kullbak-Leibler divergence
    % between P{_k}{^1} and P{_k}{^2}, where P{_k}{^i} represents the (Dirichlet-distributed) 
    % probabilities of transitioning from state k to any other state according to model i 
    % (i.e. the k-th row of the TPM)
    
    % State distribution - state is a structure array with the posterior and prior distributions of each state, so here we are storing these structure arrays for the specific k
    hs = hmm_p.state(k);
    hs0 = hmm_q.state(k);
    
    % 'hs.W.Mu_W' is empty in Neuroimage 2021 paper so all this is ignored
    % (WHAT DOES THIS MEAN? I THOUGHT IT MIGHT BE THAT WE DIDNT MODEL THE MEAN BUT WHEN I DO MODEL THE MEAN IT STILL DOESN'T GO HERE?)
%     if ~isempty(hs.W.Mu_W)
%         if train.uniqueAR || ndim==1
%             if train.uniqueAR || ndim==1
%                 D = D + nu(k) * gauss_kl(hs.W.Mu_W, hs0.W.Mu_W, hs.W.S_W, hs0.W.S_W);
%             else
%                 D = D + nu(k) * gauss_kl(hs.W.Mu_W, hs0.W.Mu_W, ...
%                     permute(hs.W.S_W,[2 3 1]), permute(hs0.W.S_W,[2 3 1]));
%             end
%         elseif strcmp(train.covtype,'diag') || ...
%                 strcmp(train.covtype,'uniquediag') || strcmp(train.covtype,'pca')
%             for n = 1:ndim
%                 D = D + nu(k) * gauss_kl(hs.W.Mu_W(Sind(:,n),n),hs0.W.Mu_W(Sind(:,n),n), ...
%                     permute(hs.W.S_W(n,Sind(:,n),Sind(:,n)),[2 3 1]),...
%                     permute(hs0.W.S_W(n,Sind(:,n),Sind(:,n)),[2 3 1]));
%             end
%         else % full or uniquefull % this is the case for Neuroimage 2021
%             mu_w = hs.W.Mu_W'; % Mu_W is the mean matrix of the MAR coefficients (although we're not using MAR??)
%             mu_w = mu_w(:);
%             mu_w0 = hs0.W.Mu_W';
%             mu_w0 = mu_w0(:);
%             D = D + nu(k) * gauss_kl(mu_w,mu_w0, hs.W.S_W, hs0.W.S_W); % S_W is the covariance matrix of the MAR coefficients (although we're not using MAR??)
%         end
%     end
    
    switch train.covtype
%         case 'diag'
%             for n=1:ndim
%                 if ~regressed(n), continue; end
%                 D = D + nu(k) * gamma_kl(hs.Omega.Gam_shape,hs0.Omega.Gam_shape, ...
%                     hs.Omega.Gam_rate(n),hs0.Omega.Gam_rate(n));
%             end
        case 'full' % this is the case for Neuroimage 2021 - but is it? we shouldn't do wishart KL here?
            try % 'try' 'catch' basically just say that if there is ever an error in any of this then jump to the catch statement (kind of like iferror() in excel)
                D = D + nu(k) * wishart_kl(hs.Omega.Gam_rate(regressed,regressed),...
                    hs0.Omega.Gam_rate(regressed,regressed), ...
                    hs.Omega.Gam_shape,hs0.Omega.Gam_shape);
            catch
                error(['Error computing kullback-leibler divergence of the cov matrix - ' ...
                    'Something strange with the data?'])
            end            
    end
    
%     if ~isempty(orders) && ~train.uniqueAR && ndim>1
%         for n1=1:ndim
%             for n2=1:ndim
%                 if (train.symmetricprior && n2<n1) || S(n1,n2)==0, continue; end
%                 D = D + nu(k) * gamma_kl(hs.sigma.Gam_shape(n1,n2),hs0.sigma.Gam_shape(n1,n2), ...
%                     hs.sigma.Gam_rate(n1,n2),hs0.sigma.Gam_rate(n1,n2));
%             end
%         end
%     end
%     if ~isempty(orders)
%         for i=1:length(orders)
%             D = D + nu(k) * gamma_kl(hs.alpha.Gam_shape,hs0.alpha.Gam_shape, ...
%                 hs.alpha.Gam_rate(i),hs0.alpha.Gam_rate(i));
%         end
%     end
end

end

function nu = compute_nu (Pi,P) % nu is the greek letter that looks like a v, see p11 from Neuroimage 2021 paper
% Pi = initial state probabilities (greek letter pi in paper)
% P = transition probability matrices
eps = 1e-6; % we want to find the limit as n goes to infinity, so we set this to be super small to mimic this
nu = Pi * P; % we start with pi^1 * (P^1)^1 i.e. pi*P, then we want to continue to multiply this by (P^1)^2,(P^1)^3,...,(P^1)^n etc, and find the limit as n tends to infinity
while true
    % instead of a 'do...while' loop, we can mimic this behaviour using 'while true'
    % so all this while true loop is doing is checking the is condition
    % below, and finishing the loop once it becomes false
    % i.e., is mean(xxx)<eps? If mean(xxx) is bigger than eps, then we continue, as soon as it is, we are done
    nu0 = nu; 
    nu = nu * P;
    if mean(nu(:)-nu0(:))<eps, break; end
end  
end
