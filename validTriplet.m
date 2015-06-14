function valid = validTriplet(triplet, D, view)
% VALIDTRIPLET checks if the triplet is uniquely satisfied
%
% valid = validTriplet(triplet, D, view)
%
% Function validTriplet checks if the triplet is uniquely satisfied in the
% corresponding view, based on the pairwise distance matrix D.
%
% input arguments:
% triplet   ----  triplet to check
% D         ----  pairwiese distance matrix
% view      ----  corresponding view to check
%
% output arguments:
% valid     ----  binary indicator variable, true if the triplet is unique
%
% (C) Ehsan Amid, Aalto University
%
% Reference:
% E. Amid, A. Ukkonen, "Multiview Triplet Embedding: Learning Attributes in
% Multiple Maps", in International Conference on Machine Learning (ICML), 2015.

M = size(D,3); % number of views
label = zeros(M-1,1);

cnt = 1;
for m = 1:M
    if m == view
        continue
    end
    label(cnt) = D(triplet(1),triplet(2),m) < D(triplet(1),triplet(3),m);
    cnt = cnt + 1;
end

valid = ~any(label);