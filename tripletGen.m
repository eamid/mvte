function triplets = tripletGen(X, num_const)
% TRIPLETGEN generates synthetic triplets
%
% triplets = tripletGen(X, num_const)
%
% Function tripletGen generates synthetic triplets based on the features in
% X. For each query item, it randomly selects the inlier item  among the K-nearest
% neighbors of the item and the outlier item from those located far away.
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

cnt = 1;
for m = 1:M
    idx = knnsearch(X{m},X{m},'K',N);
    for n = 1:N
        id1 = n;
        for t = 1:num_const
            id2 = randi(K-1) + 1;
            id3 = min(N, id2 + round(N/2) + randi(N));
            triplets(cnt, :) = [id1 idx(n,[id2 id3])];
            cnt = cnt + 1;
        end
    end
end
