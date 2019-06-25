function [points, probabilities, K] = ...
      generate_world(side_length, latent_mean, length_scale, ...
                     output_scale, num_points, random_seed)

  % seed rng for reproducible results, by default use matlab's
  % default seed
  if (nargin < 6)
    random_seed = 'default';
  end
  rng(random_seed);

  % generate grid points
  [x, y] = meshgrid(1:side_length, 1:side_length);
  points = [x(:), y(:)];
  points = points(randsample(side_length*side_length, num_points), :);

  % latent mean
  m = latent_mean * ones(size(points, 1), 1);

  % latent covariance
  hyperparameters = [log(length_scale); log(output_scale)];
  K = covSEiso(hyperparameters, points);

  K = (K + K') / 2;

  % sample latent function and squash through sigmoid
  probabilities = normcdf(mvnrnd(m, K))';

end