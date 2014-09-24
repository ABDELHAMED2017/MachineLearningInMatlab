% For each of the 100 training cases (x,y), make predictions 
% using the 50 training cases. MSE on the test set?

test_len = length(x_test);
train_len = length(x_train);

predicted_y_vector = zeros(1,test_len);

MSE = 0;

for i=1:test_len
   
    [predicted_y, predicted_y_index] = NearestNeighbor1(x_train, y_train, train_len, x_test(i));
    
    predicted_y_vector(i) = predicted_y;
    
    MSE = MSE + (predicted_y - y_test(i))^2;
    
end

MSE = MSE / test_len; %mean

%plot the training data:
figure;
plot(x_train, y_train, 'ro');
title('Training data');
%plot the testing data:
figure; 
plot(x_test, y_test, 'bo');
title('Test data');
%plot the predicted:
figure; 
plot(x_test, predicted_y_vector, 'ko');
title('Predicted data');

disp(strcat('Mean Squared Error is:  ',num2str(MSE)));