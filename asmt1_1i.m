%%%%%%%%%%%%%%%%%%%%%%%%%%% ENSEMBLE %%%%%%%%%%%%%%%%%%%%%%%%%%
% (aka put together all that messy code from before)
% VARIABLE CONVENTIONS:
% y_predict_X
% MSE_X 
% where X = exercise model corresponding to an experiment, e.g. 1a ==
% nearest neighbor.


%%%%%%%%%%%%%%%%%%%%%%%%%%% 1.a - Nearest Neighbor %%%%%%%%%%%%%%%%%%%%%%%%
val_len = length(x_val);
train_len = length(x_train);
y_predict_1a = zeros(1,val_len);
MSE_1a = 0;
for i=1:val_len
    [predict_y, predict_y_index] = NearestNeighbor1(x_train, y_train, train_len, x_test(i));
    y_predict_1a(i) = predict_y;
    MSE_1a = MSE_1a + (predict_y - y_test(i))^2; 
end
MSE_1a = MSE_1a / val_len; %mean

%%%%%%%%%%%%%%%%%%%%%%%%%% 1.b - K Nearest Neighbor %%%%%%%%%%%%%%%%%%%%%%%
test_len = length(x_test);
val_len = length(x_val);
train_len = length(x_train);
y_predict_1b = zeros(1,test_len);
MSE_1b = 0;
min_k = 1;
min_MSE = 9999999999999999; %large arbitrary number which definitely won't be a min MSE
for k=1:10
    for i=1:val_len %for validation data
        predict_y = NearestNeighborK(x_train, y_train, train_len, x_val(i), k);
        MSE_1b = MSE_1b + (predict_y - y_val(i))^2;
    end
    MSE_1b = MSE_1b / val_len;
    disp(strcat('Mean Squared Error over validation is:  ',num2str(MSE_1b),' for k=',num2str(k)));
    if MSE_1b < min_MSE
       min_k = k;
       min_MSE = MSE_1b;
    end
    %Reset MSE for next k iteration:
    MSE_1b = 0;
end
for i=1:test_len   
    predict_y = NearestNeighborK(x_train, y_train, train_len, x_test(i), min_k); %k
    y_predict_1b(i) = predict_y;
    MSE_1b = MSE_1b + (predict_y - y_test(i))^2;
end
MSE_1b = MSE_1b / test_len; %mean

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 1.c - Linear Regression (2 param)%%%%%%%%%%
train_len = length(x_train);
X_train = [ones(train_len,1), x_train']; %in workspace, x_train saved as row vector
[w, error_vector] = LinearRegression(X_train, y_train', 0.01, -0.1, 0.1, 200000, 0, 0);
% Make predictions for the test data:
test_len = length(x_test);
X_test = [ones(test_len,1), x_test']; %create the features for test data
y_predict_1c = X_test*w; %compute the prediction for test data
MSE_1c = mean((y_predict_1c-y_test').^2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 1.d - Linear Regression (k params) %%%%%%%%
train_len = length(x_train);
xx = x_train'; %in workspace, x_train saved as row vector
% Form features - polynomial powers of training data
X_train = [ones(train_len,1)]; 
w_all = cell(10);
MSE_times = cell(10);
% Create the 10 models over the training data:
for k=1:10
    X_train = [X_train, normalizeVector(xx.^k)]; %extend previous design matrix by adding new polynomial feature
    if (k > 1)
       disp(num2str(size(X_train))); 
    end
    [w, error_vector, MSE_1d, MSE_time] = LinearRegression(X_train, normalizeVector(y_train'), 0.001, -0.01, 0.01, 10000, -Inf, 0);
    w_all{k} = w; %save the model params
    MSE_times{k} = MSE_time;
end;
% For each model, test over the validation data and get their MSEs:
val_len = length(x_val);
xx = x_val'; %in workspace, x_val saved as row vector
X_val = [ones(val_len,1)];
min_MSE = Inf;
k_best = 1;
w_best = Inf;
for k=1:10
    X_val = [X_val, normalizeVector(xx.^k)]; %extend previous design matrix by adding new polynomial feature
    y_predict_1d = X_val*w_all{k}; %compute the prediction for validation data
    y_predict_ls = X_val*w_ls_test{k};
    MSE_1d = mean((y_predict_1d-y_val').^2);
    MSE_ls = mean((y_predict_ls-y_val').^2);
    if MSE_1d < min_MSE
       min_MSE = MSE_1d;
       k_best = k;
       w_best = w_all{k};
    end
end
% Use the best model to fit the test data:
test_len = length(x_test);
xx = x_test'; %in workspace, x_val saved as row vector
X_test = [ones(test_len,1)];
for k=1:k_best
    X_test = [X_test, normalizeVector(xx.^k)]; %extend previous design matrix by adding new polynomial feature
end
y_predict_1d = X_test*w_best;
MSE_1d = mean((y_predict_1d-normalizeVector(y_test')).^2);
%"unnormalize": z = (x-u)/s ==> x = z*s + u
stdy = std(y_test);
meany = mean(y_test); 
y_predict_1d = stdy*y_predict_1d + meany;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 1.e - Neural Network %%%%%%%%%%%%%%%%%%%%%%
adj_matrix = zeros(8,8);
adj_matrix(1,3:8) = ones(1,6);
adj_matrix(2,3:7) = ones(1,5);
adj_matrix(3:7,8) = ones(5,1);
xx = zeros(length(x_train_t),2);
xx(:,1) = 1;
xx(:,2) = x_train_t;
num_itrs = 10000;
MSE_time = zeros(num_itrs, 10);
t = 1:num_itrs;
MSE_at_itrs = zeros(1,20);
MSE_min = Inf;
lr_best = 0;
itr_best = 1;
neurons_best = 0;
for i=1:20
    learning_rate = 0.03/(2^i);
    [y_predict_1e, mse_over_time, neurons] = NeuralNetwork(adj_matrix, xx, y_train_t, 2, 5, 1, learning_rate, num_itrs);
    MSE_time(:,i) = mse_over_time;
    %plot(t,mse_over_time);    
    MSE_1e = mse_over_time(num_itrs);
    MSE_at_itrs(i) = MSE_1e;
    disp(strcat('MSE is::',num2str(MSE_1e)));
    disp(strcat('for learning_rate == ',num2str(learning_rate)));
    if (MSE_1e < MSE_min)
        MSE_min = MSE_1e;
        lr_best = learning_rate;
        neurons_best = neurons;
        itr_best = i;
    end
end
% Use the best model (by learning rate) to predict for test data:
xx = zeros(length(x_test_t),2);
xx(:,1) = 1;
xx(:,2) = x_test_t;
[y_predict_1e, MSE_1e] = NeuralNetworkEval(adj_matrix, xx, y_test_t, 2, 5, 1, neurons_best);

%%%%%%%%%%%%%%%%%%%%%% 1.f - Neural Networks (Random Restarts) %%%%%%%%%%%%
adj_matrix = zeros(8,8);
adj_matrix(1,3:8) = ones(1,6);
adj_matrix(2,3:7) = ones(1,5);
adj_matrix(3:7,8) = ones(5,1);
xx = zeros(length(x_train_t),2);
xx(:,1) = 1;
xx(:,2) = x_train_t;
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
    [y_predict_1f, mse_over_time, neurons] = NeuralNetwork(adj_matrix, xx, y_train_t, 2, 5, 1, learning_rate, num_itrs);
    MSE_time(:,i) = mse_over_time;
    %plot(t,mse_over_time);
    
    MSE_1f = mse_over_time(num_itrs);
    MSE_at_itrs(i) = MSE_1f;
    disp(strcat('MSE is::',num2str(MSE_1f)));
    disp(strcat('for learning_rate == ',num2str(learning_rate)));
    if (MSE_1f < MSE_min)
        MSE_min = MSE_1f;
        lr_best = learning_rate;
        neurons_best = neurons;
        itr_best = i;
    end
end
xx = zeros(length(x_test_t),2);
xx(:,1) = 1;
xx(:,2) = x_test_t;
[y_predict_1f, MSE_1f] = NeuralNetworkEval(adj_matrix, xx, y_test_t, 2, 5, 1, neurons_best);

%%%%%%%%%%%%%%%%%%% 1.g - Neural Networks (Random Restarts, k hidden) %%%%%
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
        [y_predict_1g, mse_over_time, neurons] = NeuralNetwork(adj_matrix, xx, y_val_t, 2, k, 1, learning_rate, num_itrs);
        MSE_1g = mse_over_time(num_itrs);
        MSE_over_ks(i,k) = MSE_1g;
        disp(strcat('MSE is::',num2str(MSE_1g)));
        disp(strcat('for k == ',num2str(k)));
        if (MSE_1g < MSE_min(k))
            MSE_min(k) = MSE_1g;
            neurons_best{k} = neurons;
            itr_best(k) = i;
            if (MSE_1g < MSE_min_all)
                k_best = k;
                neurons_best_all = neurons;
            end
        end
    end
end
xx = zeros(length(x_test_t),2);
xx(:,1) = 1;
xx(:,2) = x_test_t;
k = k_best;
adj_matrix = zeros(3+k,3+k);
adj_matrix(1,3:3+k) = ones(1,k+1);
adj_matrix(2,3:2+k) = ones(1,k); %ones(1,k)
adj_matrix(3:2+k,8) = ones(k,1);
[y_predict_1g, MSE_1g] = NeuralNetworkEval(adj_matrix, xx, y_test_t, 2, k, 1, neurons_best_all);

%%%%%%%%%%%%%%%%%% 1.h - Neural Networks (Early Stopping) %%%%%%%%%%%%%%%%%
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
[y_predict_1h, mse_over_time_val, neurons_all] = NeuralNetworkEarlyStop(adj_matrix, xx, y_val_t, 2, k, 1, learning_rate, num_itrs);
[MSE_val,min_index] = min(mse_over_time_val);
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
[y_predict_1h, MSE_1h] = NeuralNetworkEval(adj_matrix, xx, y_test_t, 2, k, 1, neurons_best_all);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CONSOLIDATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Averaging all models:
y_predict_final_1 = y_predict_1a + y_predict_1b + y_predict_1c + y_predict_1d + ...
                  y_predict_1e + y_predict_1f + y_predict_1g + y_predict_1h;
y_predict_final_1 = y_predict_final_1 / 8;
MSE_final_1 = mean((y_predict_final_1-y_test_t).^2);
figure; plot(x_test_t, y_test_t, 'yo', x_test_t, y_predict_final_1, 'r+');

% Averaging top m models:
MSE_vector = [MSE_1a, MSE_1b, MSE_1c, MSE_1d, MSE_1e, MSE_1f, MSE_1g, MSE_1h];
y_predict_list = [y_predict_1a, y_predict_1b, y_predict_1c, y_predict_1d, ...
                  y_predict_1e, y_predict_1f, y_predict_1g, y_predict_1h];
[s, sorted_indices] = sort(MSE_vector);
m = 4;
y_predict_final_2 = zeros(length(y_predict_final_1),1);
for i=1:m
   y_predict_final_2 = y_predict_final_2 + y_predict_list(:,sorted_indices(i)); 
end
y_predict_final_2 = y_predict_final_2 / m;
MSE_final_2 = mean((y_predict_final_2-y_test_t).^2);
figure; plot(x_test_t, y_test_t, 'yo', x_test_t, y_predict_final_2, 'r+');

% Weighted average for all models:
% Let a be a cut-off factor for the minimum MSE so that 1/MSE+a doesn't
% blow up for very low MSE models:
a = 0.001;
MSE_weight_vector = [1/(MSE_1a+a), 1/(MSE_1b+a), 1/(MSE_1c+a), 1/(MSE_1d+a), ...
                     1/(MSE_1e+a), 1/(MSE_1f+a), 1/(MSE_1g+a), 1/(MSE_1h+a)];
MSE_weight_vector = MSE_weight_vector/sum(MSE_weight_vector); %normalize sum of weights to 1
y_predict_list = [y_predict_1a, y_predict_1b, y_predict_1c, y_predict_1d, ...
                  y_predict_1e, y_predict_1f, y_predict_1g, y_predict_1h];
m = 8;
y_predict_final_3 = zeros(length(y_predict_final_1),1);
for i=1:m
   y_predict_final_3 = y_predict_final_3 + MSE_weight_vector(i)*y_predict_list(:,i); 
end
y_predict_final_3 = y_predict_final_3 / 8;
MSE_final_3 = mean((y_predict_final_3-y_test_t).^2);
figure; plot(x_test_t, y_test_t, 'yo', x_test_t, y_predict_final_3, 'r+');

disp('MSEs for AVERAGING, AVERAGING TOP 4, MSE WEIGHTED AVERAGING:');
disp(num2str(MSE_final_1));
disp(num2str(MSE_final_2));
disp(num2str(MSE_final_3));