clc; clear; close all;

%% Lorenz system parameters
sigma_l = 10; rho = 28; beta = 8/3;

%% Training setup
n = 10;         % number of noisy trajectories
dt = 0.01;      % time step
T_total = 10.0;       % match total prediction time
m = T_total / dt;     % now m = 1000 if dt = 0.01 number of the segment per trajectory
d = 2;
epsilon = 0.2;  % noise strength

% Clean Lorenz trajectory using ode45
base_init = [5, 5, 25];
[t_clean, sol_clean] = ode45(@(t, x) lorenz_rhs(t, x, sigma_l, rho, beta), ...
                             [0 T_total], base_init);
sol_clean = interp1(t_clean, sol_clean, linspace(0, T_total, m+1));

%Generate noisy training data
N = n * m;
z0 = zeros(N, d);
z1 = zeros(N, d);
t0 = zeros(N, 1);
t1 = zeros(N, 1);
count = 1;

for i = 1:n
    noise = epsilon * randn(size(sol_clean));
    noisy_traj = sol_clean + noise;

    for j = 1:m
        z0(count,:) = noisy_traj(j, 1:2);     % x-y start
        z1(count,:) = noisy_traj(j+1, 1:2);   % x-y end
        t0(count) = (j-1) * dt;
        t1(count) = j * dt;
        count = count + 1;
    end
end

delta = z1 - z0;

%% MOCK learning
lambda = 1e-4;
sigma = 0.5;
M = zeros(N,N);

for i = 1:N
    for j = 1:N
        M(i,j) = ((t1(i)-t0(i))*(t1(j)-t0(j))/4) * ( ...
            kernel(z0(i,:), z0(j,:), sigma) + ...
            kernel(z0(i,:), z1(j,:), sigma) + ...
            kernel(z1(i,:), z0(j,:), sigma) + ...
            kernel(z1(i,:), z1(j,:), sigma) ...
        );
    end
end

alpha = zeros(N, d);
for dim = 1:d
    alpha(:, dim) = (M + lambda * eye(N)) \ delta(:, dim);
end

%Predict trajectory using MOCK
y0 = base_init(1:2);  % x-y part only
steps = 1000;
dt_pred = 0.01;
y = zeros(steps+1, d);
y(1,:) = y0;

for k = 1:steps
    v = zeros(1, d);
    for i = 1:N
        w = (t1(i) - t0(i)) / 2 * ( ...
            kernel(y(k,:), z0(i,:), sigma) + ...
            kernel(y(k,:), z1(i,:), sigma));
        v = v + w * alpha(i,:);
    end
    y(k+1,:) = y(k,:) + dt_pred * v;
end
%
t_pred = linspace(0, steps * dt_pred, steps + 1);
[~, yt3d] = ode45(@(t, x) lorenz_rhs(t, x, sigma_l, rho, beta), t_pred, base_init);
yt = yt3d(:, 1:2);

% 2D Trajectory Plot
figure; hold on;
plot(y(:,1), y(:,2), 'r-', 'LineWidth', 2);
plot(yt(:,1), yt(:,2), 'b--', 'LineWidth', 2);
scatter(z0(:,1), z0(:,2), 10, 'ko', 'filled');
plot([z0(:,1) z1(:,1)]', [z0(:,2) z1(:,2)]', 'k-', 'LineWidth', 0.5);
legend('MOCK Predicted', 'Ground Truth', 'Train Start/End');
title('MOCK Prediction vs Ground Truth (2D Projection)');
axis equal; grid on;

% 2D Animation
figure; hold on; axis equal; grid on;
xlim([-30, 30]); ylim([-30, 30]);
title('2D Animation: Lorenz vs MOCK');
xlabel('x'); ylabel('y');
h_truth = plot(NaN, NaN, 'b--', 'LineWidth', 2);
h_mock = plot(NaN, NaN, 'r-', 'LineWidth', 2);
legend('Lorenz Truth', 'MOCK Prediction');

for k = 1:steps+1
    set(h_truth, 'XData', yt(1:k,1), 'YData', yt(1:k,2));
    set(h_mock,  'XData', y(1:k,1),  'YData', y(1:k,2));
    pause(0.01);
end

%3D Reconstruction from 2D MOCK + dz model
z_pred = zeros(steps+1, 1);
z_pred(1) = base_init(3);

for k = 1:steps
    x = y(k,1); y_ = y(k,2); z = z_pred(k);
    dz = x * y_ - beta * z;
    z_pred(k+1) = z + dt_pred * dz;
end
y3d_pred = [y, z_pred];

%3D Plot
figure;
plot3(yt3d(:,1), yt3d(:,2), yt3d(:,3), 'b--', 'LineWidth', 2); hold on;
plot3(y3d_pred(:,1), y3d_pred(:,2), y3d_pred(:,3), 'r-', 'LineWidth', 2);
grid on; xlabel('x'); ylabel('y'); zlabel('z');
legend('Lorenz Ground Truth', 'MOCK Reconstructed');
title('3D Reconstruction from 2D MOCK Trajectory');

%% 
function val = kernel(x, y, sigma)
    val = exp(-norm(x - y)^2 / (2 * sigma^2));
end

function dx = lorenz_rhs(~, x, sigma, rho, beta)
    dx = zeros(3,1);
    dx(1) = sigma * (x(2) - x(1));
    dx(2) = x(1) * (rho - x(3)) - x(2);
    dx(3) = x(1) * x(2) - beta * x(3);
end