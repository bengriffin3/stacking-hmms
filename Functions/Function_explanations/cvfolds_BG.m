function folds = cvfolds_BG(Y,CVscheme,allcs)
% allcs can be a N x 1 vector with family memberships, an (N x N) matrix
% with family relationships, or empty.
%rng('default') % set seed for reproducibility
if nargin<3, allcs = []; end % if no family structure provided then set to empty set
is_cs_matrix = (size(allcs,2) == 2); % test to see if allcs is a matrix (1) or vector (0)
% I am using twins which is a matrix structure

[N,q] = size(Y); % N = number of subjects, q = number of features to predict (I always set this to 1)

%CVscheme is one number, representing the number of folds to be used in
%this CV (e.g. 10 for 10-fold CV or 1 for LOO)
if CVscheme==0, nfolds = N; %%% I THINK THIS SHOULD SAY IF CVscheme = 1???
else nfolds = CVscheme;
end

% if no family structure is provided...
% [we provide twins so can ignore this bit)
if ~isempty(allcs)
    folds = cell(nfolds,1); % store folds in a cell array
    % if number of folds is equal to number of subjects (LOO) then just
    % set each fold to be a subject
    if nfolds==N
        for j = 1:N, folds{j} = j; end
        return
    else % else if we're using e.g. 10-fold CV
        if q > 1 % if more than 1 feature to predict
            % Y is the feature(s) we are trying to predict
            Y = nets_class_mattovec(Y);
            c = cvpartition(Y,'KFold',nfolds);
        elseif length(unique(Y)) <= 4 % if we only have 4 or fewer distinct categories for the feature
            c = cvpartition(Y,'KFold',nfolds);
        else % create 
            c = cvpartition(N,'KFold',nfolds);
        end
    end
    for k = 1:nfolds
        folds{k} = find(c.test(k));
    end
    return
end

% if only 1 variable (this is what I do) and number of distinct elements of
% a variable is less than or equal to 4 (this rarely happens except for
% sex and other binary variables and other categorical variables with
% limited options)
if q == 1 && length(unique(Y)) <= 4
    Y = nets_class_vectomat(Y);
    q = size(Y,2);
end

% q is greater than 1 if there is a multi-class variable to predict, so
% since all of the variables I am trying to predict at contiuous, q is
% always 1 and so do_stratified is always 0
%%% QUESTION: DOES THAT ACTUALLY MEAN NON STRATIFICATION IS DONE? I DON'T
%%% THINK SO
do_stratified = q > 1; % q = 1 so not do stratified
folds = cell(nfolds,1); grotDONE = false(N,1);
counts = zeros(nfolds,q); Scounts = sum(Y);
foldsDONE = false(nfolds,1); % track...
foldsUSED = false(nfolds,1); % track...
j = 1;


while j<=N % for each sbuject we want to place it in a fold
    % if subject has already been done, then set j to next subject and return back to while loop
    if grotDONE(j), j = j+1; continue; end 
    Jj = j; % select subject we want t place in a fold
    % pick up all of this family
    if is_cs_matrix % yes twins is a matrix 
        if size(find(allcs(:,1)==j),1)>0, Jj=[j allcs(allcs(:,1)==j,2)']; end % ALLCS MATRIX - find the family members of this subject
    else
        if allcs(j)>0
            Jj = find(allcs==allcs(j))'; % ALLCS VECTOR - find the family members of this subject
        end
    end; Jj = unique(Jj);
    if do_stratified % if we want to stratify
        % how many of each class there is
        if length(Jj)>1, countsI = sum(Y(Jj,:));
        else, countsI = Y(Jj,:);
        end
        % which fold is furthest from the wished class counts?
        d = -Inf(nfolds,1);
        for i = 1:nfolds
            if foldsDONE(i), continue; end
            c = counts(i,:) + countsI; 
            d(i) = sum( ( Scounts - c ) );
        end
        % to break the ties, choose the fold with less examples
        m = max(d); ii = (d==m);
        counts2 = sum(counts,2); counts2(~ii) = Inf; 
        [~,ii] = min(counts2);
        counts(ii,:) = counts(ii,:) + countsI;
    else % just choose the fold with less examples (if no stratification)
        [~,ii] = min(counts); % if q=1 (in our case), counts is just a counter of the subjects in each fold
        counts(ii) = counts(ii) + length(Jj); % do we then add one to the counter of that fold
    end
    % update folds, and the other indicators
    folds{ii} = [ folds{ii} Jj ]; % Now add the subject to the selected fold
    grotDONE(Jj) = true; % this confirms all the subjects we just placed in a fold have been completed
    if length(folds{ii}) >= N/nfolds, foldsDONE(ii) = true; end % once we have enough members in the fold then we have done the fold (e.g. 1000 subjects 10 folds, once we hit 100 subjects)
    foldsUSED(ii) = true;
    j = j+1; % start the loop again on the next subject
end

folds = folds(foldsUSED); 
% once we have loops through all subjects, we only want to keep the folds that actually contain subjects
% so for example, if we said we wanted to do LOO CV for 1000 subjects, the
% function creates 1000 folds to begin with but only uses 430 because of
% family structures, so then we only take the 430 folds we want

end

% if vector
function Ym = nets_class_vectomat(Y,classes)
N = length(Y);
if nargin<2, classes = unique(Y); end
q = length(classes);
Ym = zeros(N,q);
for j=1:q, Ym(Y==classes(j),j) = 1; end
end

% if matrix
function Y = nets_class_mattovec(Ym,classes)
if nargin<2, q = size(Ym,2); classes = 1:q; 
else q = length(classes); 
end
Y = zeros(size(Ym,1),1);
for j=1:q, Y(Ym(:,j)>.5) = classes(j); end
end
