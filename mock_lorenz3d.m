clc; clear; close all;

%% Lorenz parameters
sigma_l = 10;
rho = 28;
beta = 8/3;

% Training setup
n = 150;        % trajectories
m = 50;       %
dt = 0.01;     % 
N = n * m;
d = 2;

z0 = zeros(N, d);
z1 = zeros(N, d);
t0 = zeros(N,1);
t1 = zeros(N,1);

% Simulate and slice
count = 1;
for i = 1:n
    x0 = 30 * (rand(1,3)-0.5);
    traj = zeros(m+1, 3);
    traj(1,:) = x0;

    for j = 1:m
        x = traj(j,1); y = traj(j,2); z = traj(j,3);
        dx = sigma_l * (y - x);
        dy = x * (rho - z) - y;
        dz = x * y - beta * z;
        traj(j+1,:) = traj(j,:) + dt * [dx, dy, dz];

        z0(count,:) = traj(j,1:2);
        z1(count,:) = traj(j+1,1:2);
        t0(count) = (j-1)*dt;
        t1(count) = j*dt;
        count = count + 1;
    end
end

delta = z1 - z0;

%% Hyperparameters
lambda = 1e-3;
sigma = 1.0;

% Kernel matrix M
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

% Solve for alpha
alpha = zeros(N, d);
for dim = 1:d
    alpha(:, dim) = (M + lambda * eye(N)) \ delta(:, dim);
end

%% Predict new trajectory using MOCK
y0 = [0.5, 0.5];
T_pred = 10.0;  % longer prediction horizon
steps = round(T_pred / dt);
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
    y(k+1,:) = y(k,:) + dt * v;
end

%% Ground truth trajectory
y0_full = [y0, 25];
yt3d = zeros(steps+1, 3);
yt3d(1,:) = y0_full;
for k = 1:steps
    x = yt3d(k,1); y_ = yt3d(k,2); z = yt3d(k,3);
    dx = sigma_l * (y_ - x);
    dy = x * (rho - z) - y_;
    dz = x * y_ - beta * z;
    yt3d(k+1,:) = yt3d(k,:) + dt * [dx, dy, dz];
end
yt = yt3d(:,1:2);

%% Plot results
figure; hold on;
plot(y(:,1), y(:,2), 'r-', 'LineWidth', 2);
plot(yt(:,1), yt(:,2), 'b--', 'LineWidth', 2);
scatter(z0(:,1), z0(:,2), 5, 'ko', 'filled');
plot([z0(:,1) z1(:,1)]', [z0(:,2) z1(:,2)]', 'k-', 'LineWidth', 0.5);
legend('MOCK Predicted', 'Ground Truth', 'Training Segments');
title('MOCK Trajectory Prediction with Richer Lorenz Training');
axis equal; grid on;
%% Animation
figure;
axis equal; grid on; hold on;
xlim([-30, 30]); ylim([-30, 30]);
title('Trajectory Comparison Animation');
xlabel('x'); ylabel('y');

h_truth = plot(NaN, NaN, 'b--', 'LineWidth', 2);
h_mock = plot(NaN, NaN, 'r-', 'LineWidth', 2);
legend('Lorenz Projected', 'MOCK Predicted');

for k = 1:steps+1
    set(h_truth, 'XData', yt(1:k,1), 'YData', yt(1:k,2));
    set(h_mock, 'XData', y(1:k,1),  'YData', y(1:k,2));
    pause(0.01);
end

%% 3D Reconstruction
z_pred = zeros(steps+1,1);
z_pred(1) = 25;
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

%% Gaussian Kernel
function val = gaussian_kernel(x, y, sigma)
    val = exp(-norm(x - y)^2 / (2 * sigma^2));
end
