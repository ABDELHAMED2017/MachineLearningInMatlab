% K-nearest neighbors: 
% use the training data to make predictions for the validation data. 
% Compute and report classErr for k = 1, 3, 5, ... 11, and lowest classErr(k)
% Use this k to make predictions for the 100 test cases and report classErr

test_len = length(x_test);
val_len = length(x_val);
train_len = length(x_train);

y_predict_2b = zeros(test_len,1);

classErr = 0;
min_k = 1;
min_classErr = 9999999999999999; %large arbitrary number which definitely won't be a min classErr

for k=1:2:11
    
    for i=1:val_len %for validation data
        predicted_y = round(NearestNeighborK(x_train, y_train, train_len, x_val(i), k));
        classErr = classErr + nnz(predicted_y - y_val(i));
    end
    classErr = classErr / val_len;
    disp(strcat('Classification error over validation is:  ',num2str(classErr),' for k=',num2str(k)));
    if classErr < min_classErr
       min_k = k;
       min_classErr = classErr;
    end
    %Reset classErr for next k iteration:
    classErr = 0;
end

disp(strcat('The min classErr over validation is:  ',num2str(min_classErr),' for k=',num2str(min_k)));

%test:
%min_k = 25;

%classErr_vec = zeros(1,40);
%for k=1:40
for i=1:test_len
   
    predicted_y = round(NearestNeighborK(x_train, y_train, train_len, x_test(i), min_k)); %k
    y_predict_2b(i) = predicted_y;
   
    
end


%figure;
%plot(classErr_vec);

classErr = nnz(y_predict_2b - y_test)/length(y_test);

figure; 
plot(x_test, y_test, 'bo');
title('Test data');
%plot the predicted:
figure; 
plot(x_test, y_predict_2b, 'ko');
title('Predicted data');

disp(strcat('Classification error for test data is:  ',num2str(classErr)));