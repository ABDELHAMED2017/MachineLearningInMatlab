% K-nearest neighbors: 
% use the training data to make predictions for the validation data. 
% Compute and report MSE for k = 1, 2, ..., 10, and lowest MSE(k)
% Use this k to make predictions for the 100 test cases and report MSE

test_len = length(x_test);
val_len = length(x_val);
train_len = length(x_train);

predicted_y_vector = zeros(1,test_len);

MSE = 0;
min_k = 1;
min_MSE = 9999999999999999; %large arbitrary number which definitely won't be a min MSE

for k=1:10
    
    for i=1:val_len %for validation data
        predicted_y = NearestNeighborK(x_train, y_train, train_len, x_val(i), k);
        MSE = MSE + (predicted_y - y_val(i))^2;
    end
    
    MSE = MSE / val_len;
    disp(strcat('Mean Squared Error over validation is:  ',num2str(MSE),' for k=',num2str(k)));
    if MSE < min_MSE
       min_k = k;
       min_MSE = MSE;
    end
    %Reset MSE for next k iteration:
    MSE = 0;
end

disp(strcat('The min MSE over validation is:  ',num2str(min_MSE),' for k=',num2str(min_k)));

%test:
%min_k = 25;

%MSE_vec = zeros(1,40);
%for k=1:40
for i=1:test_len
   
    predicted_y = NearestNeighborK(x_train, y_train, train_len, x_test(i), min_k); %k
    predicted_y_vector(i) = predicted_y;
    MSE = MSE + (predicted_y - y_test(i))^2;
    
end
%    MSE = MSE/test_len;
%    MSE_vec(k) = MSE;
%    MSE = 0;
%end

%figure;
%plot(MSE_vec);

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

disp(strcat('Mean Squared Error over training data is:  ',num2str(MSE)));