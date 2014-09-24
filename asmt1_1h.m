% k = 10, and early stopping.
kmax = 10;
restarts = 1;

xx = zeros(length(x_val_t),2);
xx(:,1) = 1;
xx(:,2) = x_val_t;

num_itrs = 10000; 
MSE_over_ks = zeros(restarts, kmax);
t = 1:num_itrs;
MSE_min = zeros(kmax,1)+Inf;
MSE_min_all = Inf;
k_best = 1;
itr_best = ones(kmax,1);
neurons_best_all = 0;


k = 10;

adj_matrix = zeros(3+k,3+k);
adj_matrix(1,3:3+k) = ones(1,k+1);
adj_matrix(2,3:2+k) = ones(1,k); %ones(1,k)
adj_matrix(3:2+k,3+k) = ones(k,1);

learning_rate = 0.03/(2^3); %Best LR from before
[y_predict_NN, mse_over_time_val, neurons_all] = NeuralNetworkEarlyStop(adj_matrix, xx, y_val_t, 2, k, 1, learning_rate, num_itrs);

[MSE_val,min_index] = min(mse_over_time_val);
disp(strcat('Min MSE index over validation set: ',num2str(min_index)));
% Use the index of the best MSE to choose the neurons for evaluating:
neurons_best_all = neurons_all{min_index};

% Use the best model (by MSE) to predict for test data:
xx = zeros(length(x_test_t),2);
xx(:,1) = 1;
xx(:,2) = x_test_t;
adj_matrix = zeros(3+k,3+k);
adj_matrix(1,3:3+k) = ones(1,k+1);
adj_matrix(2,3:2+k) = ones(1,k); %ones(1,k)
adj_matrix(3:2+k,8) = ones(k,1);

[y_predict_NN, MSE] = NeuralNetworkEval(adj_matrix, xx, y_test_t, 2, k, 1, neurons_best_all);
figure; plot(x_test_t, y_test_t, 'yo', x_test_t, y_predict_NN, 'r+');
disp(strcat('Test MSE is::',num2str(MSE)));