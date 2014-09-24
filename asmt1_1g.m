% Random restarts and up to 10 hidden logistical units
kmax = 10;
restarts = 20;

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
neurons_best = cell(kmax);
neurons_best_all = 0;



for k=1:kmax %number of features
    for i=1:restarts %random restarts
    

        adj_matrix = zeros(3+k,3+k);
        adj_matrix(1,3:3+k) = ones(1,k+1);
        adj_matrix(2,3:2+k) = ones(1,k); %ones(1,k)
        adj_matrix(3:2+k,3+k) = ones(k,1);

        learning_rate = 0.03/(2^3); %Best LR from before
        [y_predict_NN, mse_over_time, neurons] = NeuralNetwork(adj_matrix, xx, y_val_t, 2, k, 1, learning_rate, num_itrs);

        MSE = mse_over_time(num_itrs);
        MSE_over_ks(i,k) = MSE;
        disp(strcat('MSE is::',num2str(MSE)));
        disp(strcat('for k == ',num2str(k)));
        if (MSE < MSE_min(k))
            MSE_min(k) = MSE;
            neurons_best{k} = neurons;
            itr_best(k) = i;
            if (MSE < MSE_min_all)
                k_best = k;
                neurons_best_all = neurons;
            end
        end
    end
end

%I need a better way to do this:
figure;
t = 1:restarts;
plot(t,MSE_over_ks(:,1),t,MSE_over_ks(:,2),t,MSE_over_ks(:,3),t,MSE_over_ks(:,4),t,MSE_over_ks(:,5),...
     t,MSE_over_ks(:,6),t,MSE_over_ks(:,7),t,MSE_over_ks(:,8),t,MSE_over_ks(:,9),t,MSE_over_ks(:,10));

% Use the best model (by MSE) to predict for test data:
xx = zeros(length(x_test_t),2);
xx(:,1) = 1;
xx(:,2) = x_test_t;
k = k_best;
adj_matrix = zeros(3+k,3+k);
adj_matrix(1,3:3+k) = ones(1,k+1);
adj_matrix(2,3:2+k) = ones(1,k); %ones(1,k)
adj_matrix(3:2+k,8) = ones(k,1);

[y_predict_NN, MSE] = NeuralNetworkEval(adj_matrix, xx, y_test_t, 2, k, 1, neurons_best_all);
figure; plot(x_test_t, y_test_t, 'yo', x_test_t, y_predict_NN, 'r+');
disp(strcat('Test MSE is::',num2str(MSE)));
disp(strcat('for k == ',num2str(k), '|| restart number == ',num2str(itr_best(k))));