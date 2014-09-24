% fit a k-param model to the training data.
% For each model, make predictions for the validation cases, get their MSEs
% Use that value of k to make predictions for the 100 test cases and report the test error rate.


train_len = length(x_train);
xx = x_train'; %in workspace, x_train saved as row vector
% Form features - polynomial powers of training data

X_train = [ones(train_len,1)]; 
w_all = cell(10,1);
MSE_times = cell(10,1);
w_ls_test = cell(10,1);

% Create the 10 models over the training data:
for k=1:10
    X_train = [X_train, normalizeVector(xx.^k)]; %extend previous design matrix by adding new polynomial feature
    if (k > 1)
       disp(num2str(size(X_train))); 
    end
    [w, error_vector, MSE, MSE_time] = LinearRegression(X_train, normalizeVector(y_train'), 0.001, -0.01, 0.01, 10000, -Inf, 0);
    w_all{k} = w; %save the model params
    MSE_times{k} = MSE_time;
    w_ls_test{k} = (X_train'*X_train)\X_train'*y_train';
end;

%TEST

% For each model, test over the validation data and get their MSEs:
val_len = length(x_val);
xx = x_val'; %in workspace, x_val saved as row vector
X_val = [ones(val_len,1)];
min_MSE = Inf;
k_best = 1;
w_best = Inf;
min_MSE_ls = Inf;
k_best_ls = 1;
w_best_ls = Inf;
for k=1:10
    X_val = [X_val, normalizeVector(xx.^k)]; %extend previous design matrix by adding new polynomial feature
    y_predicted = X_val*w_all{k}; %compute the prediction for validation data
    y_predicted_ls = X_val*w_ls_test{k};
    MSE = mean((y_predicted-y_val').^2);
    MSE_ls = mean((y_predicted_ls-y_val').^2);
    disp(strcat('Validation MSE is:: ',num2str(MSE),' for k==',num2str(k)));
    disp(strcat('Validation MSE_ls is:: ',num2str(MSE_ls),' for k==',num2str(k)));
    if MSE < min_MSE
       min_MSE = MSE;
       k_best = k;
       w_best = w_all{k};
    end
    if MSE_ls < min_MSE_ls
       min_MSE_ls = MSE_ls;
       k_best_ls = k;
       w_best_ls = w_ls_test{k};
    end
end

% Use the best model to fit the test data:
test_len = length(x_test);
xx = x_test'; %in workspace, x_val saved as row vector
X_test = [ones(test_len,1)];
for k=1:k_best
    X_test = [X_test, normalizeVector(xx.^k)]; %extend previous design matrix by adding new polynomial feature
end
y_predicted = X_test*w_best;

%Do the same for LS:
X_test = [ones(test_len,1)];
for k=1:k_best_ls
    X_test = [X_test, xx.^k]; %extend previous design matrix by adding new polynomial feature
end
y_predicted_ls = X_test*w_best_ls;


MSE = mean((y_predicted-y_test').^2);
disp('MSE is (gradient descent): ');
disp(num2str(MSE));
disp(strcat('for k==',num2str(k_best)));

MSE = mean((y_predicted_ls-y_test').^2);
disp('MSE is (least squares): ');
disp(num2str(MSE));
disp(strcat('for k==',num2str(k_best_ls)));

figure;
plot(x_test, y_test, 'ro');
title('Test data');

%plot the predicted data:
%"unnormalize": z = (x-u)/s ==> x = z*s + u
stdy = std(y_test);
meany = mean(y_test); 
y_predicted = stdy*y_predicted + meany;
figure; 
plot(x_test, y_predicted, 'bo');
title('Predicted data - gradient descent');
%figure; 
%plot(x_test, y_predicted_ls, 'go');
%title('Predicted data - least squares');