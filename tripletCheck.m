function [Z, ratios] = tripletCheck(y, triplets)
% TRIPLETCHECK checks if the triplets are satisfied in the maps
%
% [Z, ratios] = tripletCheck(y, triplets)
%
% Function validTriplet checks if the triplets are satisfied in the maps y.
% It returns the binary indicator variables Z and the sat. ratios as output.
%
% input arguments:
% y         ----  map points (N x dim x M)
% triplets  ----  triplets to check
%
% output arguments:
% Z         ----  binary indicator variables
% ratios    ----  sat. ratios
%
% (C) Ehsan Amid, Aalto University
%
% Reference:
% E. Amid, A. Ukkonen, "Multiview Triplet Embedding: Learning Attributes in
% Multiple Maps", in International Conference on Machine Learning (ICML), 2015.

id1 = triplets(:,1);
id2 = triplets(:,2);
id3 = triplets(:,3);

N = size(y,1);
M = size(y,3);
T = size(triplets,1);
ratios = zeros(T,M);

for mm = 1:M
    D = pdist2(y(:,:,mm),y(:,:,mm));
    ratios(:,mm) = D(id1 + N * (id3 -1))./D(id1 + N * (id2 -1));
end
Z = ratios > 1;
