clc;clear;close all;
%%
% Parameters
sigma = 10;
rho = 28;
beta = 8/3;
dt_sim = 0.01;
T_sim = 1;
steps_sim = T_sim / dt_sim;

% Initial conditions
N = 100;
d=2;
t0 = zeros(N,1);
t1 = ones(N,1) * 0.01;
init_points = 2 * (rand(N, 3) - 0.5);  % x, y, z in [-2.5, 2.5]
z0 = init_points(:, 1:2);  % project x and y as input for MOCK
z1 = zeros(N, 2);

% Simulate short Lorenz segments
for i = 1:N
    x = init_points(i,1);
    y = init_points(i,2);
    z = init_points(i,3);

    for k = 1:round(0.5 / dt_sim)  % evolve for ~0.5s
        dx = sigma * (y - x);
        dy = x * (rho - z) - y;
        dz = x * y - beta * z;

        x = x + dt_sim * dx;
        y = y + dt_sim * dy;
        z = z + dt_sim * dz;
    end

    z1(i,:) = [x, y];  % project final x, y
end

% Compute displacement for MOCK
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
y0_full = [1.0, 1.0, 1.0];  % Initial full state
yt3d = zeros(steps+1, 3);
yt3d(1,:) = y0_full;

for k = 1:steps
    x = yt3d(k,1); y_ = yt3d(k,2); z = yt3d(k,3);
    dx = sigma * (y_ - x);
    dy = x * (rho - z) - y_;
    dz = x * y_ - beta * z;
    yt3d(k+1,:) = yt3d(k,:) + dt * [dx, dy, dz];
end

% Project 3D trajectory to 2D
yt = yt3d(:, 1:2);
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
figure;
axis equal; grid on; hold on;
xlim([-30, 30]); ylim([-30, 30]);
title('Trajectory Comparison Animation');
xlabel('x'); ylabel('y');

% Plot elements
h_truth = plot(NaN, NaN, 'b--', 'LineWidth', 2);
h_mock = plot(NaN, NaN, 'r-', 'LineWidth', 2);
legend('Lorenz Projected', 'MOCK Predicted');

for k = 1:steps+1
    set(h_truth, 'XData', yt(1:k,1), 'YData', yt(1:k,2));
    set(h_mock, 'XData', y(1:k,1),  'YData', y(1:k,2));
    pause(0.01);
end
z_pred = zeros(steps+1,1);
z_pred(1) = 25;  % assume starting z
for k = 1:steps
    x = y(k,1); y_ = y(k,2); z = z_pred(k);
    dz = x * y_ - beta * z;
    z_pred(k+1) = z + dt * dz;
end

y3d_pred = [y, z_pred];
figure;
plot3(yt3d(:,1), yt3d(:,2), yt3d(:,3), 'b--', 'LineWidth', 2); hold on;
plot3(y3d_pred(:,1), y3d_pred(:,2), y3d_pred(:,3), 'r-', 'LineWidth', 2);
grid on; xlabel('x'); ylabel('y'); zlabel('z');
legend('Lorenz Truth (3D)', 'MOCK-Reconstructed (3D)');
title('3D Reconstruction from 2D MOCK Trajectory');
%%
function val = gaussian_kernel(x, y, sigma)
    val = exp(-norm(x - y)^2 / (2 * sigma^2));
end