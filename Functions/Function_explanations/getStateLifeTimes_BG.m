function lifetimes = getStateLifeTimes_BG(Gamma,T,K,options,threshold,threshold_Gamma,do_concat)
% Computes the state life times for the state time courses, in number of time points 
%
% Gamma can be the probabilistic state time courses (time by states),
%   which can contain the probability of all states or a subset of them,
%   or the Viterbi path (time by 1). 
% In the first case, threshold_Gamma is used to define when a state is active. 
%
% The parameter threshold is used to discard visits that are too
%   short (deemed to be spurious); these are discarded when they are shorter than 
%   'threshold' time points
%
% The parameter 'options' must be the same than the one supplied to the
% hmmmar function for training. 
%
% lifetimes is then a cell (subjects/segments/trials by states), where each element is a
%   vector of life times
%
% if do_concat is specified, the lifetimes are concatenated across
% segments of data, so that lifetimes is a cell (1 by states)
% 
% Diego Vidaurre, OHBA, University of Oxford (2016)

% if nargin<3, options = struct(); options.Fs = 1; options.downsample = 1; end
% if ~isfield(options,'Fs'), options.Fs = 1; end
% if ~isfield(options,'downsample'), options.downsample = options.Fs; end
if nargin<4 || isempty(threshold), threshold = 0; end
% if nargin<5 || isempty(threshold_Gamma), threshold_Gamma = (2/3); end
% if nargin<6 || isempty(do_concat), do_concat = true; end

%%%%%%%%%%%%% Sort time point of session out
% is_vpath = (size(Gamma,2)==1 && all(rem(Gamma,1)==0)); 
% if iscell(T)
%     if size(T,1) == 1, T = T'; end
%     for i = 1:length(T)
%         if size(T{i},1)==1, T{i} = T{i}'; end
%     end
%     Nsubj = length(T);
%     trials2subjects = zeros(length(cell2mat(T)),1); ii = 1; 
%     for i = 1:length(T)
%         Ntrials = length(T{i});
%         trials2subjects(ii:ii+Ntrials-1) = i;
%         ii = ii + Ntrials;
%     end
%     T = cell2mat(T);
% else 
%     Nsubj = length(T);
%     trials2subjects = 1:Nsubj;
% end
N = length(T); % number of subjects x number of sessions i.e. 1004 x 4 here
%%%%%%%%%%%%%%

% r = 1; 
% if isfield(options,'downsample') && options.downsample>0
%     r = (options.downsample/options.Fs);
% end

% If MAR is used
% if isfield(options,'order') && options.order > 0
%     T = ceil(r * T);
%     T = T - options.order; 
% elseif isfield(options,'embeddedlags') && length(options.embeddedlags) > 1
%     d1 = -min(0,options.embeddedlags(1));
%     d2 = max(0,options.embeddedlags(end));
%     T = T - (d1+d2);
%     T = ceil(r * T);
% else
%     options.order = (sum(T) - size(Gamma,1)) / length(T);
%     T = T - options.order; 
% end

% if is_vpath % viterbi path
    vpath = Gamma; 
    %K = length(unique(vpath)); % note number of states
    Gamma = zeros(length(vpath),K);
    for k = 1:K
       Gamma(vpath==k,k) = 1; % note down the time points when each state is active, the idea is then to find how many consecutive time points each state is active for
    end

% else
%     warning(['Using the Viterbi path is here recommended instead of the state ' ...
%         'probabilistic time courses (Gamma)'])
%     K = size(Gamma,2); 
%     Gamma = Gamma > threshold_Gamma;
% end

% if ~do_concat
%     lifetimes = cell(Nsubj,K);
%     for j = 1:N
%         t0 = sum(T(1:j-1));
%         ind = (1:T(j)) + t0;
%         if length(ind)==1, continue; end
%         jj = trials2subjects(j);
%         for k = 1:K
%             lifetimes{jj,k} = [lifetimes{jj,k} aux_k(Gamma(ind,k),threshold)];
%         end
%     end
% else

    lifetimes = cell(1,K); % we want the lifetimes for each state
    for j = 1:N % for each subject and session
        t0 = sum(T(1:j-1)); % the start time point of this subjects first session
        ind = (1:T(j)) + t0; % time point of this subjects first session (start to finish)
%         if length(ind)==1, continue; end
        for k = 1:K
            lifetimes{k} = [lifetimes{k} aux_k(Gamma(ind,k),threshold)];
        end
    end
% end



end


function lifetimes = aux_k(g,threshold)

lifetimes = [];

while ~isempty(g)
    
    t = find(g==1,1); % for this state, find the time point the state is next active
    if isempty(t), break; end % if it is empty (i.e. we have got to the end) then we 'break' out of the 'while' loop
    tend = find(g(t+1:end)==0,1); % for this state, find the next time point that it is NOT active
    
    if isempty(tend) % end of trial % i.e. if the state is active until the end of the trial, then note this down and 'break' out the 'while' loop
%         if (length(g)-t+1)>threshold
            lifetimes = [lifetimes (length(g)-t+1)]; 
%         else
%             g(t:end) = 0; % too short visit to consider
%         end
        break;
    end
    
%     if tend<threshold % too short visit, resetting g and trying again
%         g(t:t+tend-1) = 0;
%         continue;
%     end
    
    lifetimes = [lifetimes tend];
    tnext = t + tend;
    g = g(tnext:end);
    
end

end
