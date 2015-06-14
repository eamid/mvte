% This is a working example of the MVTE algorithm. For further details,
% please see the paper.
%
% (C) Ehsan Amid, Aalto University
%
% Reference:
% E. Amid, A. Ukkonen, "Multiview Triplet Embedding: Learning Attributes in
% Multiple Maps", in International Conference on Machine Learning (ICML), 2015.

%% Load data
load objects_xo.mat
% generate a cell array of features
X{1} = X_col; % color features
X{2} = X_shape; % shape features

%% Generate triplets
num_const = 50; % number of triplets per instance
triplets = tripletGenUnique(X,num_const);

%% MVTE algorithm
M = 2; % number of views
dim = 2; % number of dimensions
y = mvte(triplets, M, dim); % find the maps

%%  Plot the results
close all
figure
subplot(1,2,1)
scatter(y(idx_shape==1,1,1),y(idx_shape==1,2,1),90,col(idx_shape==1,:),'x','lineWidth',2.5)
hold on
scatter(y(idx_shape==2,1,1),y(idx_shape==2,2,1),90,col(idx_shape==2,:),'o','lineWidth',2.5)
title('MVTE - First Map','fontsize',20)
axis square
set(gcf,'position',[100 220 1100 400])
set(gca, 'XTickLabelMode', 'manual', 'XTickLabel', []);
set(gca, 'YTickLabelMode', 'manual', 'YTickLabel', []);
colormap(spring)

subplot(1,2,2)
scatter(y(idx_shape==1,1,2),y(idx_shape==1,2,2),90,col(idx_shape==1,:),'x','lineWidth',2.5)
hold on
scatter(y(idx_shape==2,1,2),y(idx_shape==2,2,2),90,col(idx_shape==2,:),'o','lineWidth',2.5)
title('MVTE - Second Map','fontsize',20)
 axis square
 set(gcf,'position',[100 220 1100 400])
set(gca, 'XTickLabelMode', 'manual', 'XTickLabel', []);
set(gca, 'YTickLabelMode', 'manual', 'YTickLabel', []);
colormap(spring)
