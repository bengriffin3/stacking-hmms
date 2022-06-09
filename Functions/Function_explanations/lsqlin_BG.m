function [X,resnorm,residual,exitflag,output,lambda] = lsqlin_BG(C,d,A,b,Aeq,beq,lb,ub,X0,options,varargin)
%LSQLIN Constrained linear least squares.
%   X = LSQLIN(C,d,A,b) attempts to solve the least-squares problem
%
%           min  0.5*(NORM(C*x-d)).^2       subject to    A*x <= b
%            x
%
%   where C is m-by-n.
%
%   X = LSQLIN(C,d,A,b,Aeq,beq) solves the least-squares
%   (with equality constraints) problem:
%
%           min  0.5*(NORM(C*x-d)).^2    subject to
%            x                               A*x <= b and Aeq*x = beq
%
%   X = LSQLIN(C,d,A,b,Aeq,beq,LB,UB) defines a set of lower and upper
%   bounds on the design variables, X, so that the solution
%   is in the range LB <= X <= UB. Use empty matrices for
%   LB and UB if no bounds exist. Set LB(i) = -Inf if X(i) is unbounded
%   below; set UB(i) = Inf if X(i) is unbounded above.
%
%   X = LSQLIN(C,d,A,b,Aeq,beq,LB,UB,X0) sets the starting point to X0.
%
%   X = LSQLIN(C,d,A,b,Aeq,beq,LB,UB,X0,OPTIONS) minimizes with the default
%   optimization parameters replaced by values in OPTIONS, an argument
%   created with the OPTIMOPTIONS function. See OPTIMOPTIONS for details.
%
%   X = LSQLIN(PROBLEM) solves the least squares problem defined in
%   PROBLEM. PROBLEM is a structure with the matrix 'C' in PROBLEM.C, the
%   vector 'd' in PROBLEM.d, the linear inequality constraints in
%   PROBLEM.Aineq and PROBLEM.bineq, the linear equality constraints in
%   PROBLEM.Aeq and PROBLEM.beq, the lower bounds in PROBLEM.lb, the upper
%   bounds in PROBLEM.ub, the start point in PROBLEM.x0, the options
%   structure in PROBLEM.options, and solver name 'lsqlin' in
%   PROBLEM.solver. Use this syntax to solve at the command line a problem
%   exported from OPTIMTOOL.
%
%   [X,RESNORM] = LSQLIN(C,d,A,b) returns the value of the squared 2-norm
%   of the residual: norm(C*X-d)^2.
%
%   [X,RESNORM,RESIDUAL] = LSQLIN(C,d,A,b) returns the residual: C*X-d.
%
%   [X,RESNORM,RESIDUAL,EXITFLAG] = LSQLIN(C,d,A,b) returns an EXITFLAG
%   that describes the exit condition. Possible values of EXITFLAG and the
%   corresponding exit conditions are
%
%     0  Maximum number of iterations exceeded.
%     1  LSQLIN converged to a solution X.
%     2  Solver stalled at feasible point.
%     3  Change in the residual smaller that the specified tolerance.
%    -2  Problem is infeasible.
%    -4  Ill-conditioning prevents further optimization.
%    -8  Unable to compute step direction; no further progress can be made.
%
%   [X,RESNORM,RESIDUAL,EXITFLAG,OUTPUT] = LSQLIN(C,d,A,b) returns a
%   structure OUTPUT with the number of iterations taken in
%   OUTPUT.iterations, the type of algorithm used in OUTPUT.algorithm, the
%   number of conjugate gradient iterations (if used) in OUTPUT.cgiterations,
%   a measure of first order optimality (large-scale algorithm only) in
%   OUTPUT.firstorderopt, and the exit message in OUTPUT.message.
%
%   [X,RESNORM,RESIDUAL,EXITFLAG,OUTPUT,LAMBDA] = LSQLIN(C,d,A,b) returns
%   the set of Lagrangian multipliers LAMBDA, at the solution:
%   LAMBDA.ineqlin for the linear inequalities C, LAMBDA.eqlin for the
%   linear equalities Ceq, LAMBDA.lower for LB, and LAMBDA.upper for UB.
% 
%   [WSOUT,FVAL,EXITFLAG,...] = LSQLIN(C,d,A,b,Aeq,beq,LB,UB,WS) starts
%   lsqlin from the data in the warm start object WS, using the options
%   in WS. The returned argument WSOUT contains the solution point in
%   WSOUT.X. By using WSOUT as the initial warm start object in a
%   subsequent solver call, lsqlin can work faster.
%
%   See also QUADPROG, OPTIMWARMSTART

%   Copyright 1990-2020 The MathWorks, Inc.

% if nargin == 0
%     error(message('optimlib:lsqlin:NotEnoughInputs'))
% end

defaultopt = struct( ...
    'Algorithm','interior-point', ...
    'Diagnostics','off', ...
    'Display','final', ...
    'JacobMult',[], ...
    'LargeScale','on', ...
    'MaxIter',200, ...
    'MaxPCGIter','max(1,floor(numberOfVariables/2))', ...
    'PrecondBandWidth',0, ...
    'ProblemdefOptions', struct, ...
    'TolCon',1e-8, ...
    'TolFun',1e-8, ...
    'TolX', 1e-12, ...
    'TolFunValue',100*eps, ...
    'TolPCG',0.1, ...
    'TypicalX','ones(numberOfVariables,1)', ...
    'LinearSolver', 'auto', ...
    'ObjectiveLimit', -1e20 ...
    );


% % If just 'defaults' passed in, return the default options in X
% if nargin==1 && nargout <= 1 && strcmpi(C,'defaults')
%     X = defaultopt;
%     return;
% end

% % Handle missing arguments
% if nargin < 10
%     options = [];
%     if nargin < 9
%         X0 = [];
%         if nargin < 8
%             ub = [];
%             if nargin < 7
%                 lb = [];
%                 if nargin < 6
%                     beq = [];
%                     if nargin < 5
%                         Aeq = [];
%                         if nargin < 4
%                             b = [];
%                             if nargin < 3
%                                 A = [];
%                             end
%                         end
%                     end
%                 end
%             end
%         end
%     end
% end

% if (nargin >= 9 && isa(X0, 'optim.warmstart.LsqlinWarmStart'))
%     useWarmStart = true;
%     if ~isempty(options)
%        warning(message('optimlib:warmstart:WarmStartOptionsIgnored'));
%     end
%     options = X0.Options;
% else
%     useWarmStart = false;
% end

% % Detect problem structure input
% if nargin == 1
%     if isa(C,'struct')
%         [C,d,A,b,Aeq,beq,lb,ub,X0,options] = separateOptimStruct(C);
%     else % Single input and non-structure.
%         error(message('optimlib:lsqlin:InputArg'));
%     end
% end

% % Check for non-double inputs
% if useWarmStart
%     msg = isoptimargdbl('LSQLIN', {'C','d','A','b','Aeq','beq','LB','UB','X0'}, ...
%         C,  d,  A,  b,  Aeq,  beq,  lb,  ub,  X0.X);
% else
%     msg = isoptimargdbl('LSQLIN', {'C','d','A','b','Aeq','beq','LB','UB','X0'}, ...
%         C,  d,  A,  b,  Aeq,  beq,  lb,  ub,  X0);
% end
% 
% if ~isempty(msg)
%     error('optimlib:lsqlin:NonDoubleInput',msg);
% end

% Set up the options
if isempty(options)
    % No options passed. Set options directly to defaultopt
    options = defaultopt;
    % Set flag to optimoptions since this is a required input
    optimgetFlag = 'optimoptions';
else
    % Check for optimoptions input. When optimoptions are input, we don't need
    % to check defaultopts since optimoptions contain values for all options.
    % Also, we don't need to convert strings to characters. Optimget can just
    % read the value from the struct.
    if isa(options,'optim.options.SolverOptions')
        optimgetFlag = 'optimoptions';
    elseif isstruct(options)
        optimgetFlag = 'fast';
    else
        error('optimlib:lsqlin:InvalidOptions', ...
            getString(message('optimlib:commonMsgs:InvalidOptions')));
    end
    
    % Prepare the options for the solver
    options = prepareOptionsForSolver(options, 'lsqlin');
end

% % Options setup
% if strcmpi(optimgetFlag,'optimoptions')
%     largescale = true;
% else
%     largescale = strcmpi(optimget(options,'LargeScale',defaultopt,optimgetFlag),'on');
% end
% diagnostics = strcmpi(optimget(options,'Diagnostics',defaultopt,optimgetFlag),'on');


if nargout > 5
    computeLambda = true;
else
    computeLambda = false;
end
if nargout > 4
    computeFirstOrderOpt = true;
else
    computeFirstOrderOpt = false;
end

% Set up constant strings
trustRegion = 'trust-region-reflective';
unconstrained = 'mldivide';
interiorPoint = 'interior-point';
activeSet = 'active-set';
output.iterations = []; % initialize so that it will be the first field

% Read Algorithm
output.algorithm = optimget(options,'Algorithm',defaultopt,optimgetFlag);
if ~any(strcmpi(output.algorithm, {trustRegion,interiorPoint,activeSet}))
    error(message('optimlib:lsqlin:InvalidAlgorithm'));
end

% % Conflicting options Algorithm='trust-region-reflective' and
% % LargeScale='off'. Choose interior-point algorithm.
% % Warn later, not in case of early termination
% algAndLargeScaleConflict = strcmpi(output.algorithm,trustRegion) && ~largescale;

% % Used for trust-region-reflective algorithm.
% % Any time JacobMult is set, we need to check for naming conflicts
% % regardless of which algorithm we are using.
% mtxmpy = optimget(options,'JacobMult',defaultopt,optimgetFlag);
% % Check if name clash
% functionNameClashCheck('JacobMult',mtxmpy,'atamult','optimlib:lsqlin:JacobMultNameClash');
% 
% % Use internal Jacobian-multiply function if user does not provide JacobMult function
% if isempty(mtxmpy)
%     mtxmpy = @atamult;
% end

display = optimget(options,'Display',defaultopt,optimgetFlag);
detailedExitMsg = contains(display,'detailed');
% switch display
% case {'off', 'none'}
    verbosity = 0;
% case {'iter','iter-detailed'}
%    verbosity = 2;
% case {'final','final-detailed'}
%    verbosity = 1;
% case 'testing'
%    verbosity = 3;
% otherwise
%    verbosity = 1;
% end


% Set the constraints up: defaults and check size
[~,numberOfVariables]=size(A);

% if isempty(C) || isempty(d)
%     error(message('optimlib:lsqlin:FirstTwoArgsEmpty'))
% else
%     numberOfVariables = max([size(C,2),numberOfVariables]); % In case A is empty
% end
% [rows,cols]=size(C);
% 
% if length(d) ~= rows
%     error(message('optimlib:lsqlin:InvalidCAndD'))
% end
% 
% if length(b) ~= size(A,1)
%     error(message('optimlib:lsqlin:InvalidAAndB'))
% end
% 
% if length(beq) ~= size(Aeq,1)
%     error(message('optimlib:lsqlin:InvalidAeqAndBeq'))
% end
% 
% if ( ~isempty(A)) && (size(A,2) ~= cols)
%     error(message('optimlib:lsqlin:CAndA'))
% end
% 
% if ( ~isempty(Aeq)) && (size(Aeq,2) ~= cols)
%     error(message('optimlib:lsqlin:CAndAeq'))
% end

if isempty(X0) && ~any(strcmpi(output.algorithm,{activeSet, interiorPoint}))
    % This zero-valued X0 will potentially be changed in sllsbox or qpsub.
    % (This potentially temporary zero-valued x0 needed here for backwards
    % compatibility because it's returned in output x if early termination
    % occurs when bounds are inconsistent.)
    X0 = zeros(numberOfVariables,1);
    params.emptyInitialPoint = true;  % parameter passed to sllsbox
else
    params.emptyInitialPoint = false; % parameter passed to sllsbox
end
if isempty(A)
    A = zeros(0,numberOfVariables);
end
if isempty(b)
    b = zeros(0,1);
end
% if isempty(Aeq)
%     Aeq = zeros(0,numberOfVariables);
% end
% if isempty(beq)
%     beq = zeros(0,1);
% end

% % Set d, b and X to be column vectors
% d = d(:);
% b = b(:);
% beq = beq(:);
% X0 = X0(:);

% [X0,lb,ub,msg] = checkbounds(X0,lb,ub,numberOfVariables);
% if ~isempty(msg)
%     exitflag = -2;
%     [resnorm,residual,lambda]=deal([]);
%     output.iterations = 0;
%     output.algorithm = ''; % not known at this stage
%     output.firstorderopt=[];
%     output.cgiterations =[];
%     output.linearsolver = [];
%     output.message = msg;
%     X=X0;
%     if verbosity > 0
%         disp(msg)
%     end
%     return
% end
% 
% % Test if C is all zeros
% if (vecnorm(C(:)) == 0)
%     if ~strcmpi(output.algorithm, interiorPoint)
%         C = [];
%     else
%         % C must be a sparse matrix of correct size to prevent an error
%         % in the interior-point QP
%         C = sparse(numel(d), numberOfVariables);
%     end
% end
% 
% hasLinearConstr = ~isempty(beq) || ~isempty(b);

% Test for constraints
% if ~hasLinearConstr && all(isinf([lb;ub]))
%     output.algorithm = unconstrained;
% elseif strcmpi(output.algorithm,trustRegion)
%     if algAndLargeScaleConflict
%         warning(message('optimlib:lsqlin:AlgAndLargeScaleConflict'));
%         output.algorithm = interiorPoint;
%     elseif (rows < cols)
%         error(message('optimlib:lsqlin:MoreColsThanRows'));
%     elseif hasLinearConstr
%         warning(message('optimlib:lsqlin:LinConstraints'));
%         output.algorithm = interiorPoint;
%     end
% end

% if diagnostics
%     % Do diagnostics on information so far
%     gradflag = []; hessflag = []; constflag = false; gradconstflag = false;
%     non_eq=0;non_ineq=0; lin_eq=size(Aeq,1); lin_ineq=size(A,1);
%     XOUT=ones(numberOfVariables,1); funfcn{1} = []; confcn{1}=[];
%     diagnose('lsqlin',output,gradflag,hessflag,constflag,gradconstflag,...
%         XOUT,non_eq,non_ineq,lin_eq,lin_ineq,lb,ub,funfcn,confcn);
% end


switch output.algorithm
%     case unconstrained        
%         % Call the "no warn" mldivide to avoid disabling and re-enabling the warning
%         
%         if useWarmStart
%             flags.isLeastSquaresUnconstrained = true;
%             X = asqpdenseWarmStart(C, d, [], [], [], [], [], [], X0, flags);
%         else
%             X = matlab.internal.math.nowarn.mldivide(C,d);
%         end
%             
%         exitflag = 1;
%         if computeFirstOrderOpt || computeLambda
%             lambda.lower = [];
%             lambda.upper = [];
%             lambda.ineqlin = [];
%             lambda.eqlin = [];
%             output.iterations = 0;
%             output.firstorderopt = [];
%             output.constrviolation = [];
%             output.cgiterations = [];
%             output.linearsolver = [];
%             output.message = '';
%         end
        
%     case trustRegion
%         params.verb = verbosity; % pack parameters together into struct
%         defaultopt.TolFun = 100*eps;
%         [X,resnorm,residual,firstorderopt,iterations,cgiterations,exitflag,lambda,msg]=...
%             sllsbox(C,d,lb,ub,X0,params,options,defaultopt,mtxmpy,computeLambda,varargin{:});
%         output.iterations = iterations;
%         output.firstorderopt = firstorderopt;
%         output.cgiterations = cgiterations;
%         output.constrviolation = [];
%         output.linearsolver = [];
%         output.message = msg;
        
%%%%%% Interior point is default algorithm %%%%%%%%%%%%%%%%%%%
    case interiorPoint
        
%         % Set ConvexCheck flag to notify QP solver that problem is convex.
        options.ConvexCheck = 'off';
        
        % If the output structure is requested, we must reconstruct the
        % Lagrange multipliers in the postsolve. Therefore, set computeLambda
        % to true if the output structure is requested.
        flags.computeLambda = computeFirstOrderOpt;
        flags.detailedExitMsg = detailedExitMsg;
        flags.verbosity = verbosity;
        flags.caller = 'lsqlin';
        
%         % If user specifies trust-region-reflective algorithm is switched to
%         % interior-point for some reason, we need to change X0 back to empty if
%         % it was before so we don't output a redundent message.
%         if (params.emptyInitialPoint)
%             X0 = [];
%         end
        
%         thisMaxIter = optimget(options,'MaxIter',defaultopt,optimgetFlag);
%         if ischar(thisMaxIter)
%             error('optimlib:lsqlin:InvalidMaxIter', ...
%                 getString(message('MATLAB:optimfun:optimoptioncheckfield:notANonNegInteger','MaxIter')));
%         end
%         
        % compute for later use in algorithm
        f = -C'*d;
        %C(1:10,:);
        %d(1:10,:);
        
        % Check which solver the user requested
        linearSolver = optimget(options,'LinearSolver',defaultopt,optimgetFlag);
        autoSelect = strcmp(linearSolver, 'auto');


        % Full QP
        if (autoSelect && ~issparse(C)) || strcmp(linearSolver, 'dense')
            %C = full(C);
            H = C'*C;
            [X, ~, exitflag, output, lambda] = ...
                ipqpdense(H, f, A, b, Aeq, beq, lb, ub, X0, flags, ...
                options, defaultopt, varargin{:});
            output.linearsolver = 'dense';
            
%         % Sparse QP
%         else
%             % For sparse C, often, it's best to form H using dense operations.
%             % Check for truly sparse (and large) matrices before envoking
%             % sparse product.
%             if size(C,1) >= 200 && (nnz(C)/numel(C) <= 0.03)
%                 C = sparse(C);
%                 H = C'*C;
%             else % Compute dense product
%                 C = full(C);
%                 H = sparse(C'*C);
%             end
%             [X, ~, exitflag, output, lambda] = ...
%                 ipqpcommon(H, f, A, b, Aeq, beq, lb, ub, X0, flags, ...
%                 options, defaultopt, varargin{:});
%             output.linearsolver = 'sparse';
        end
        
%         output.algorithm = interiorPoint; % overwrite "interior-point-convex"
%         output.cgiterations = [];
%         if isempty(lambda)
%             X = []; residual = []; resnorm = [];
%             return;
%         end
        
        % Presolve may have removed variables and constraints from the problem.
        % Postsolve will re-insert the primal and dual solutions after the main
        % algorithm has run. Therefore, constraint violation and first-order
        % optimality must be re-computed (below).
        
%     otherwise % must be active-set due to error checking above
%         
%         % If the output structure is requested, we must reconstruct the
%         % Lagrange multipliers in the postsolve. Therefore, set computeLambda
%         % to true if the output structure is requested.
%         flags.computeLambda = computeFirstOrderOpt;
%         flags.detailedExitMsg = detailedExitMsg;
%         flags.verbosity = verbosity;
%         flags.caller = 'lsqlin';
%         
%         % Check for empty which ==> all-zero C
%         if ~isempty(C)
%             C = full(C);
%             H = C'*C;
%             f = -C'*d;
%         else
%             C = zeros(numel(d),numberOfVariables);
%             H = zeros(numberOfVariables);
%             f = zeros(numberOfVariables,1);
%         end
%         
%         mConstr = numel(b) + numel(beq) + sum(isfinite(lb)) + sum(isfinite(ub));
% 
%         if useWarmStart
%             % Re-use info from previous solve to try and reduce iteration
%             % count.
%             flags.isLeastSquaresUnconstrained = (mConstr == 0);
%             [X, ~, exitflag, output, lambda] = ...
%                 asqpdenseWarmStart(H, f, A, b, Aeq, beq, lb, ub, X0, flags);
%         else
%             % Set default options
%             defaultopt.MaxIter = 10*(numberOfVariables + mConstr);
% 
%             options.MaxIter = optimget(options, 'MaxIter', defaultopt, optimgetFlag);
%             if strcmpi(options.MaxIter, '10*(numberOfVariables + numberOfConstraints)')
%                 options.MaxIter = defaultopt.MaxIter;
%             end
%             defaultopt.TolX = 1e-8;
%             options.ConvexCheck = 'off'; % we have a guarentee that these problems are convex
% 
%             [X, ~, exitflag, output, lambda] = ...
%                 asqpdense(H, f, A, b, Aeq, beq, lb, ub, X0, flags, options, defaultopt);
%         end
end

% if any(strcmpi(output.algorithm , {interiorPoint, activeSet, unconstrained}))
%     
% %     if useWarmStart
% %         xsol = X.X; % weird syntax to avoid copy
% %     else
% %         xsol = X;
% %     end
%     
%     if (nargout >= 2)
%         residual = C*xsol-d;
%         resnorm = sum(residual.*residual);
%     end
%     
%     % Compute constraint violation if the output structure is requested
%     % Compute first order optimality if needed
%     
%     % For the interior-point algorithm, if no initial point was provided by
%     % the user and the presolve has declared the problem infeasible or
%     % unbounded, X will be empty. The lambda structure will also be empty,
%     % so do not compute constraint violation or first-order optimality if
%     % lambda is missing.
%     if computeFirstOrderOpt && strcmpi(output.algorithm, interiorPoint)
%         output.firstorderopt = computeKKTErrorForQPLP(H,f,A,b,Aeq,beq,lb,ub,lambda,xsol);
%         output.constrviolation = norm([Aeq*xsol-beq;max([A*xsol - b;xsol - ub;lb - xsol],0)],Inf);
%     end
%     output.cgiterations = [];
% end
