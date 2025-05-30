clc; clear; close all;

%% Parameters
n = 10;     % number of trajectories
m = 10;     % segments per trajectory
d = 2;
dt = 0.05;  % time step per segment
N = n * m;  % total training segments

z0 = zeros(N, d);
z1 = zeros(N, d);
t0 = zeros(N,1);
t1 = zeros(N,1);

% Define dynamics
f_dyn = @(x, y) [-y + 0.1*x*(1 - x^2 - y^2); ...
                  x + 0.1*y*(1 - x^2 - y^2)];

%% Generate multiple trajectories and extract segments
count = 1;
for i = 1:n
    theta0 = 2 * pi * rand();
    r = 1.0 + 0.2 * randn();  % small perturbation
    x0 = [r * cos(theta0), r * sin(theta0)];
    traj = zeros(m+1, d);
    traj(1,:) = x0;

    for j = 1:m
        x = traj(j,1); y = traj(j,2);
        dxdy = f_dyn(x, y);
        traj(j+1,:) = traj(j,:) + dt * dxdy';

        z0(count,:) = traj(j,:);
        z1(count,:) = traj(j+1,:);
        t0(count) = (j-1)*dt;
        t1(count) = j*dt;
        count = count + 1;
    end
end

delta = z1 - z0;

%% Hyperparameters
lambda = 1e-3;
sigma = 0.5;

% Precompute kernel matrix M
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

%% Solve for alpha
alpha = zeros(N, d);
for dim = 1:d
    rhs = delta(:, dim);
    alpha(:, dim) = (M + lambda * eye(N)) \ rhs;
end

%% Predict new trajectory using MOCK
y0 = [0.5, 0.5];
dt_pred = 0.01;
T_pred = 5.0;
steps = round(T_pred / dt_pred);
y = zeros(steps+1, d);
y(1,:) = y0;

for k = 1:steps
    v = zeros(1, d);
    for i = 1:N
        w = (t1(i) - t0(i))/2 * ( ...
            gaussian_kernel(y(k,:), z0(i,:), sigma) + ...
            gaussian_kernel(y(k,:), z1(i,:), sigma));
        v = v + w * alpha(i,:);
    end
    y(k+1,:) = y(k,:) + dt_pred * v;
end

%% Ground truth trajectory for validation
yt = zeros(steps+1, d);
yt(1,:) = y0;
for k = 1:steps
    dxdy = f_dyn(yt(k,1), yt(k,2));
    yt(k+1,:) = yt(k,:) + dt_pred * dxdy';
end

%% Plot
figure; hold on;
plot(y(:,1), y(:,2), 'r-', 'LineWidth', 2);
plot(yt(:,1), yt(:,2), 'b--', 'LineWidth', 2);
scatter(z0(:,1), z0(:,2), 'ko', 'filled');
scatter(z1(:,1), z1(:,2), 'gx');
plot([z0(:,1) z1(:,1)]', [z0(:,2) z1(:,2)]', 'k--');
legend('MOCK Predicted', 'Ground Truth', 'Train Start', 'Train End');
title('MOCK on Rotational + Radial Dynamics');
axis equal; grid on;

%% Gaussian Kernel
function val = gaussian_kernel(x, y, sigma)
    val = exp(-norm(x - y)^2 / (2 * sigma^2));
end
