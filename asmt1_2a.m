% For each of the 100 training cases (x,y), make predictions 
% using the 50 training cases. MSE on the test set?

test_len = length(x_test);
train_len = length(x_train);

y_predict_2a = zeros(test_len,1);

classErr = 0;

for i=1:test_len
   
    [predicted_y, predicted_y_index] = NearestNeighbor1(x_train, y_train, train_len, x_test(i));
    
    y_predict_2a(i) = predicted_y;
    
    
end

classErr = nnz(y_predict_2a - y_test)/length(y_test);

plot(x_test, y_test, 'bo');
title('Test data');
%plot the predicted:
figure; 
plot(x_test, y_predict_2a, 'ko');
title('Predicted data');

disp(strcat('Classification Error is:  ',num2str(classErr)));