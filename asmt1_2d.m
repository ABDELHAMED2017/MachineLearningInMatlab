% Linear regression classification - 
% fit a 2-param model to the training data:

% Create design matrix X:
% Each row is a training case, representing (x0=1, x1=x) in a 2-param model
% where we will try to fit the model y(t) = w0*x0(t) + w1*x1(t)

train_len = length(x_train);

X_train = [ones(train_len,1), x_train]; %in workspace, x_train saved as row vector

[w, error_vector] = LinearRegression(X_train, y_train, 0.01, -0.1, 0.1, 300000, 0, 0);

% Make predictions for the test data:

test_len = length(x_test);
X_test = [ones(test_len,1), x_test]; %create the features for test data

y_predict_2d = X_test*w; %compute the prediction for test data

classErr = nnz(round(y_predict_2d) - y_test)/length(y_test);
disp('classErr is: ');
disp(num2str(classErr));

threshErrors = zeros(100,1);
for i=1:100
   y_predict_thresh_test = binaryThresholdVector(y_predict_2d,i/100); 
   threshErrors(i) = nnz(round(y_predict_thresh_test) - y_test)/length(y_test);
end
figure; plot(threshErrors);

figure;
plot(x_test, y_test, 'ro');
title('Test data');
%plot the predicted data:
figure; 
plot(x_test, y_predict_2d, 'bo');
title('Predicted data');
figure; 
plot(x_test, round(y_predict_2d), 'ko');
title('Predicted data - thresholded');