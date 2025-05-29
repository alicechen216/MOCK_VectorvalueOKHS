clc;clear;close all;
%%
% Classical rotational dynamics: dx/dt = -y, dy/dt = x
% Number of training samples
N = 100;
d = 2;

% Time interval
t0 = zeros(N,1);
t1 = ones(N,1) * 0.05;  % Short time window

% Initial positions on a circle (or arbitrary)
theta = linspace(0, 2*pi, N+1);
theta(end) = [];  % remove duplicate point
r = 1.0;  % radius
z0 = [r * cos(theta)', r * sin(theta)'];

% Compute dynamics for small time step using Euler integration
delta_t = t1(1) - t0(1);
z1 = zeros(size(z0));
for i = 1:N
    x = z0(i,1); y = z0(i,2);
    dx = -y; dy = x;
    z1(i, :) = [x + dx * delta_t, y + dy * delta_t];
end

% delta for training
delta = z1 - z0;

%%
% Hyperparameters
lambda = 1e-3;
sigma = 0.5;

% Precompute M
M = zeros(N,N);
for i = 1:N
    for j = 1:N
        M(i,j) = ((t1(i)-t0(i))*(t1(j)-t0(j))/4) * ( ...
            gaussian_kernel(z0(i,:), z0(j,:), sigma) + ...
            gaussian_kernel(z0(i,:), z1(j,:), sigma) + ...
            gaussian_kernel(z1(i,:), z0(j,:), sigma) + ...
            gaussian_kernel(z1(i,:), z1(j,:), sigma) ...
        );
    end
end
%%
alpha = zeros(N, d);
for dim = 1:d
    rhs = delta(:, dim);
    alpha(:, dim) = (M + lambda * N * eye(N)) \ rhs;
end
%%
% Initial condition (new test point)
y0 = [0.5, 0.5];  % test point in 2D

% Time settings
dt = 0.01;
T = 1.0;
steps = round(T / dt);
y = zeros(steps+1, d);
y(1,:) = y0;

% Forward integration (Euler method)
for k = 1:steps
    v = zeros(1, d);  % velocity vector from learned vector field

    for i = 1:N
        weight = (t1(i)-t0(i))/2 * ( ...
            gaussian_kernel(y(k,:), z0(i,:), sigma) + ...
            gaussian_kernel(y(k,:), z1(i,:), sigma) ...
        );
        v = v + weight * alpha(i, :);
    end

    y(k+1,:) = y(k,:) + dt * v;
end
%%
% Simulate ground truth trajectory using same initial condition
yt = zeros(steps+1, d);
yt(1,:) = y0;

for k = 1:steps
    x = yt(k,1);
    y_ = yt(k,2);
    dx = -y_;
    dy = x;
    yt(k+1,:) = yt(k,:) + dt * [dx, dy];
end

% Plot predicted vs ground truth
figure; hold on;

% MOCK predicted trajectory
plot(y(:,1), y(:,2), 'r-', 'LineWidth', 2);

% Ground truth trajectory
plot(yt(:,1), yt(:,2), 'b--', 'LineWidth', 2);

% Training data
scatter(z0(:,1), z0(:,2), 'ko', 'filled');
scatter(z1(:,1), z1(:,2), 'gx');
plot([z0(:,1) z1(:,1)]', [z0(:,2) z1(:,2)]', 'k--');

legend('MOCK Predicted', 'Ground Truth', 'Train Start', 'Train End');
title('MOCK Inference vs Ground Truth Trajectory');
axis equal; grid on;
%%
function val = gaussian_kernel(x, y, sigma)
    val = exp(-norm(x - y)^2 / (2 * sigma^2));
end