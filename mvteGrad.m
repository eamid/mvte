function gradY = mvteGrad(y, triplets, num, z, P)
% MVTEGRAD calculates the gradient in the MVTE algorithm
%
% gradY = mvteGrad(y, triplets, num, z, P)
%
% Function mvteGrad calculates the gradient w.r.t the map points y.
%
% input arguments:
% y         ----  map points (N x dim x M)
% triplets  ----  matrix of triplets (T x 3), acquired on N items
% num       ----  constant factor in the gradient (see mvte.m)
% z         ----  indicator variables
% P         ----  matrix of probabilities
%
% output arguments:
% gradY     ----  gradient
%
% (C) Ehsan Amid, Aalto University
%
% Reference:
% E. Amid, A. Ukkonen, "Multiview Triplet Embedding: Learning Attributes in
% Multiple Maps", in International Conference on Machine Learning (ICML), 2015.


[N, dim, M] = size(y);
id1 = triplets(:,1);
id2 = triplets(:,2);
id3 = triplets(:,3);

gradY = zeros(N,dim,M);

for m = 1:M
    ym = y(:,:,m);
    gradY1 = -2 * bsxfun(@times, (ym(id1, :) - ym(id2, :)), z(:,m) .* (1 - P(:,m)) .* num(id1 + (id2-1)*N + (m-1)*N^2))...
             +2 * bsxfun(@times, (ym(id1, :) - ym(id3, :)), z(:,m) .* (1 - P(:,m)) .* num(id1 + (id3-1)*N + (m-1)*N^2));
    gradY2 =  2 * bsxfun(@times, (ym(id1, :) - ym(id2, :)), z(:,m) .* (1 - P(:,m)) .* num(id1 + (id2-1)*N + (m-1)*N^2));
    gradY3 = -2 * bsxfun(@times, (ym(id1, :) - ym(id3, :)), z(:,m) .* (1 - P(:,m)) .* num(id1 + (id3-1)*N + (m-1)*N^2));
    
    for d = 1:dim
        gradY(:,d,m) = accumarray([id1;id2;id3], [gradY1(:,d); gradY2(:,d); gradY3(:,d)],[N 1]);
    end
end
