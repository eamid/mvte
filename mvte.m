function y = mvte(triplets, M, dim, w)
% MVTE performs multiview triplet embedding algorithm
%
% y = mvte(triplets, M, dim, w)
%
% Function mvte applies the multiview triplet embedding algorithm on the
% input triplets. M specifies the number views and dim indicates the number
% of dimensions.
%
%
% input arguments:
% triplets  ----  matrix of triplets (T x 3), acquired on N items
% M         ----  number of views (default = 2)
% dim       ----  number of dimensions of the maps (default = 2)
% w         ----  tail-heaviness parameter: 0 for mvte (default), 1 for t-mvte
%
% output arguments:
% y         ----  output maps (N x dim x M)
%
% (C) Ehsan Amid, Aalto University
%
% Reference:
% E. Amid, A. Ukkonen, "Multiview Triplet Embedding: Learning Attributes in
% Multiple Maps", in International Conference on Machine Learning (ICML), 2015.


if ~exist('M','var') || isempty(M)
    M = 2;
end

if ~exist('dim','var') || isempty(dim)
    dim = 2;
end

if ~exist('w','var') || isempty(w)
    w = 0;
elseif ~(w == 0 || w == 1)
    error('invalid tail-heaviness parameter')
end

tol = 1e-7;
lrateY = 1;  % learning rate for y
numItr = 1000; % maximum number of iterations

id1 = triplets(:,1);
id2 = triplets(:,2);
id3 = triplets(:,3);


N = max(triplets(:)); % number of items
T = size(triplets,1); % number of triplets

y = 0.0001 * randn(N,dim,M); % initial output values

D = zeros(N,N,M); % pairwise distance matrix

for m = 1:M
    D(:,:,m) = pdist2(y(:,:,m),y(:,:,m));
end

if w == 0
    num = exp(-D.^2); % MVTE
else
    num = 1./(1+D.^2); % t-MVTE
end

nuIdx = bsxfun(@plus,(1:N+1:N^2)',0:N^2:(M-1)*N^2);
num(nuIdx(:)) = 0;

pair1 = num(bsxfun(@plus, id1 + N * (id2-1), 0:N^2:(M-1)*N^2));
pair2 = num(bsxfun(@plus, id1 + N * (id3-1), 0:N^2:(M-1)*N^2));
P = pair1./max(realmin,(pair1+pair2)); % probabilities

C_b = Inf; % initial cost
y_best = y; % best y so far

[~,ratio] = tripletCheck(y,triplets); % find sat. ratios
ratio(ratio<=1) = eps;
ratio(isinf(ratio)) = 1e5;
z = bsxfun(@rdivide,ratio,sum(ratio,2));

no_increase = 0;
for iter = 1:numItr
    if no_increase > 100
        break
    end
    if w == 0
        gradY = mvteGrad(y, triplets, ones(N,N,M), z, P);
    else
        gradY = mvteGrad(y, triplets, num, z, P);
    end
    y = y + lrateY * gradY / T * N;
    
    for m = 1:M
        D(:,:,m) = pdist2(y(:,:,m),y(:,:,m));
    end
    if w == 0
        num = exp(-D.^2);
    else
        num = 1./(1+D.^2);
    end
    nuIdx = bsxfun(@plus,(1:N+1:N^2)',0:N^2:(M-1)*N^2);
    num(nuIdx(:)) = 0;
    
    pair1 = num(bsxfun(@plus, id1 + N * (id2-1), 0:N^2:(M-1)*N^2));
    pair2 = num(bsxfun(@plus, id1 + N * (id3-1), 0:N^2:(M-1)*N^2));
    P = pair1./max(realmin,(pair1+pair2)); % update probabilities
    [~,ratio] = tripletCheck(y,triplets);
    ratio(ratio<=1) = eps;
    ratio(isinf(ratio)) = 1e5;
    z = bsxfun(@rdivide,ratio,sum(ratio,2)); % update indicators
    
    C = -sum(z(:).*log(max(P(:),realmin))); % cost
    if C + tol < C_b
        lrateY = lrateY * 1.1; % increase learning rate
        y_best = y;
        no_increase = 0; % decrease learning rate
    else
        lrateY = lrateY/2;
        no_increase = no_increase + 1;
    end
    C_b = C; % update cost
    
    if ~rem(iter,10)
        fprintf('iteration %4d/%d, error = %6.4f\n',iter,numItr,C);
    end
end

y = y_best; % return y
