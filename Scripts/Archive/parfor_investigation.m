

N = 1000;

parfor r = 1:5
    for d = 1:5
        for n1 = 1:N
            for n2 = n1+1:N

                DistHMM(n1,n2,r,d) = 5*6

               % (hmm_kl_BG(out.HMMs_dualregr{n1},out.HMMs_dualregr{n2}) ...
               %     + hmm_kl(out.HMMs_dualregr{n2},out.HMMs_dualregr{n1}))/2;
                %DistHMM(n2,n1,r,d) = DistHMM(n1,n2,r,d);

            end
        end
    end
end

%%
clc
N = 1001;
%DistHMM = NaN(N,N,rep,dir);
tic
DistHMM = cell(rep,1);
parfor r = 1:5
    DistHMM{r} = zeros(N,N,dir);
    for d = 1:5
        for n1 = 1:N
            for n2 = n1+1:N
 
                    DistHMM{r}(n1,n2,d) = 5*6


            end
        end
    end
end
toc
%%
clc
x = rand(10,10,3);
x_new = x - bsxfun(@times, eye(size(x(:,:,1))), x) % set diagonal elements to 0

bsxfun(@minus, tril(x(:,:,1)), x)

for i = 1:3
    x_new(:,:,i) = x_new(:,:,i) - tril(x_new(:,:,i)); 
end
x_new;

y_new = x_new;
for i = 1:3
    y_new(:,:,i) = y_new(:,:,i) + (triu(x_new(:,:,i)))';
end
y_new
    % z = y'
% y_new = y + z


%%
s = 1001;
n = 100;
tic
a = cell(s,1);
parfor k = 1:s
   a{k} = zeros(n);
   for i = 1:n
      for j=i+1:n
         a{k}(i,j) = rand; %or whatever scalar is returned by your function
      end
   end
end
toc




