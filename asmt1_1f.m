% Random restarts

adj_matrix = zeros(8,8);
adj_matrix(1,3:8) = ones(1,6);
adj_matrix(2,3:7) = ones(1,5);
adj_matrix(3:7,8) = ones(5,1);
xx = zeros(length(x_train_t),2);
xx(:,1) = 1;
xx(:,2) = x_train_t;

%y = NeuralNetwork(adj_matrix, input, targets, num_input_nodes, num_hidden_nodes, ...
%                   num_output_nodes, learning_rate, num_itrs)

% log errors over time:
num_itrs = 10000;
MSE_time = zeros(num_itrs, 10);
t = 1:num_itrs;
MSE_at_itrs = zeros(1,20);
MSE_min = Inf;
lr_best = 0;
itr_best = 1;
neurons_best = 0;

for i=1:20

    learning_rate = 0.03/(2^3); %Best LR from before
    [y_predict_NN, mse_over_time, neurons] = NeuralNetwork(adj_matrix, xx, y_train_t, 2, 5, 1, learning_rate, num_itrs);
    MSE_time(:,i) = mse_over_time;
    %plot(t,mse_over_time);
    
    MSE = mse_over_time(num_itrs);
    MSE_at_itrs(i) = MSE;
    disp(strcat('MSE is::',num2str(MSE)));
    disp(strcat('for learning_rate == ',num2str(learning_rate)));
    if (MSE < MSE_min)
        MSE_min = MSE;
        lr_best = learning_rate;
        neurons_best = neurons;
        itr_best = i;
    end
end

%I need a better way to do this:
figure;
t = log10(t);
plot(t,MSE_time(:,1),t,MSE_time(:,2),t,MSE_time(:,3),t,MSE_time(:,4),t,MSE_time(:,5),...
     t,MSE_time(:,6),t,MSE_time(:,7),t,MSE_time(:,8),t,MSE_time(:,9),t,MSE_time(:,10));

% Use the best model (by MSE) to predict for test data:
xx = zeros(length(x_test_t),2);
xx(:,1) = 1;
xx(:,2) = x_test_t;
[y_predict_NN, MSE] = NeuralNetworkEval(adj_matrix, xx, y_test_t, 2, 5, 1, neurons_best);
figure; plot(x_test_t, y_test_t, 'yo', x_test_t, y_predict_NN, 'r+');
disp(strcat('Test MSE is::',num2str(MSE)));
disp(strcat('for learning_rate == ',num2str(lr_best)));
figure; hist(MSE_at_itrs,25);