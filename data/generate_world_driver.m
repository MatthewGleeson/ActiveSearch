
function [labels_random, labels_deterministic,problem]= generate_world_driver()

addpath(genpath('./gpml'))
side_length = 50;
latent_mean = -1;
length_scale = 3;
output_scale = 1;
num_points = 2500;
[points, probabilities, K] = ...
  generate_world(side_length, latent_mean, length_scale, output_scale, num_points);

problem.points = points;
problem.num_points = size(points, 1);
problem.num_classes = 2;

labels_random = rand(size(probabilities)) < probabilities/6;

close all
% figure; imagesc(reshape(probabilities, side_length, side_length));
% figure; imagesc(reshape(labels_random, side_length, side_length));


labels_deterministic = probabilities > 0.7;
% figure; imagesc(reshape(labels_deterministic, side_length, side_length));

%output = problem
end



%dataname = sprintf('toy_problem.mat');
%save(dataname, 'labels_random', 'labels_deterministic', 'problem');

