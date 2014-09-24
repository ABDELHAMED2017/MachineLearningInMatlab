% K-nearest neighbors: 
% use the training data to make predictions for the validation data. 
% Compute and report classErr for k = 1, 3, 5, ... 11, and lowest classErr(k)
% Use this k to make predictions for the 100 test cases and report classErr

%%%%%%%%%%%%%%%%%%%% Methodology: %%%%%%%%%%%%%%%%%%%%%%%%%
%
%For this problem, we must associate with each model, call them k, a (log-)
%likelihood function, l(k). Our goal in this maximum likelihood learning will 
%then be to select the model k that maximizes the likelihood function of the data. Start with the likelihood function L(k): 
%
%L(k) = Product(P(y(t) | x(t), k)) for training/validation cases t = 1, …, T
%
%For binary labels, the probability of y = 1 for a given x, k (input and model) is given by:
%
%p = Pr(y = 1 | x, k) = C1 / (C0 + C1)
%and 1 –p = Pr(y = 0 | x, k) = C0 / (C0 + C1)
%
%Where C0 and C1 are the neighbor counts of nearby 1s and 0s, plus the additional pseudocounts of 0.1.
%
%The probability of any y at test case t (each y having a binary label) can be written as one expression:
%
%P(y(t)| x(t), k) = [p(t)^y(t)]*[(1-p(t))^(1-y(t))]
%
%Which is a convenient closed-form expression analogous to that in logistical regression. Computationally, 
%our log-likelihood function can be expressed as:
%
%l(k) = log(Product(P(y(t) | x(t), k))) 
%l(k) = sum(y(t)*log(p(t)) + (1-y(t))*log(1-p(t))), the sum extending over all training cases t = 1, …, T
%
%The NearestNeighborsK function was modified to be NearestNeighborsKprobability to help with this task, 
%and the script in asmt1_2c implemented the calculating and comparison of l(k) for k = 1,3,…,11.


test_len = length(x_test);
val_len = length(x_val);
train_len = length(x_train);

y_predict_2c = zeros(test_len,1);

H = nnz(y_train);
T = numel(y_train)-H;

log_likelihood = 0;
best_k = 1;
max_LL = -9999999999999999; %large arbitrary -VE number which definitely won't be a max LL
min_classErr = 9999999999999999;
classErr = 0;

for k=1:2:11
    
    for i=1:val_len %for validation data    
        p_y_1 = NearestNeighborKprobability(x_train, y_train, train_len, x_val(i), k);
        % returned as p(y(t)=1|x(t),k)
        p_y_0 = 1 - p_y_1;
        
        % Update metrics:
        log_likelihood = log_likelihood + (y_val(i)*log(p_y_1)+(1-y_val(i))*log(p_y_0));
        classErr = classErr + nnz(round(p_y_1) - y_val(i));
    end
    classErr = classErr / val_len;
    
    disp(strcat('Log likelihood over validation is::',num2str(log_likelihood),'|classErr is::',num2str(classErr),' for k=',num2str(k)));
    if log_likelihood > max_LL
       best_k = k;
       max_LL = log_likelihood;
    end
    %Reset classErr and LL for next k iteration:
    log_likelihood = 0;
    classErr = 0;
end

disp(strcat('The max LL over validation is:  ',num2str(max_LL),' for k=',num2str(best_k)));

%test:
%min_k = 25;

%classErr_vec = zeros(1,40);
%for k=1:40
numZeros = 0.1; %start with pseudocounts of 0.1
numOnes = 0.1;
classErr = 0;
for i=1:test_len
   
    p_y_1 = NearestNeighborKprobability(x_train, y_train, train_len, x_test(i), best_k); %k
    p_y_0 = 1 - p_y_1;
    y_predict_2c(i) = round(p_y_1);
    
    % Update metrics:
    log_likelihood = log_likelihood + (y_test(i)*log(p_y_1)+(1-y_test(i))*log(p_y_0));
end
classErr = nnz(y_predict_2c - y_test)/length(y_test);

figure; 
plot(x_test, y_test, 'bo');
title('Test data');
%plot the predicted:
figure; 
plot(x_test, y_predict_2c, 'ko');
title('Predicted data');

disp(strcat('Log likelihood for test data is:  ',num2str(log_likelihood)));
disp(strcat('Classification error for test data is:  ',num2str(classErr)));