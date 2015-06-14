function triplets = tripletGenUnique(X, num_const)
% TRIPLETGENUNIQUE generates unique synthetic triplets
%
% triplets = tripletGenUnique(X, num_const)
%
% Function tripletGenUnique generates synthetic triplets based on the 
% features in X such that each triplet is only satisfied in one of the 
% feature spaces. For each query item, tripletGenUnique randomly selects the 
% inlier item  among the K-nearest neighbors of the item and the outlier 
% item from those located far away. Note that the function might get stuck 
% in the while loop.
%
% input arguments:
% X         ----  cell array of feature matrices (length = M)
% num_const ----  number of triplets per item
%
% output arguments:
% triplets  ----  output triplets
%
% (C) Ehsan Amid, Aalto University
%
% Reference:
% E. Amid, A. Ukkonen, "Multiview Triplet Embedding: Learning Attributes in
% Multiple Maps", in International Conference on Machine Learning (ICML), 2015.

M = length(X); % number of views
N = size(X{1},1); % number of items
T = N * num_const * M; % number of triplets
triplets = zeros(T,3); % initialze
K = 20; % K nearest neighbors

idx = zeros(N,N,M);
D = zeros(N,N,M); % pairwise distance matrix

for m = 1:M
    idx(:,:,m) = knnsearch(X{m},X{m},'K',N);
    D(:,:,m) = pdist2(X{m},X{m});
end

cnt = 1;
for m = 1:M
    for n = 1:N
        id1 = n;
        t = 1;
        while t <=num_const
            id2 = randi(K-1) + 1;
            id3 = min(N, id2 + round(N/2) + randi(N));
            elem = [id1 idx(n,[id2 id3],m)];
            if validTriplet(elem,D,m)
                triplets(cnt, :) = elem;
                t = t + 1;
                cnt = cnt + 1;
            end
        end
    end
end
