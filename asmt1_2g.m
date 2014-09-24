%%%%%%%%%%%%%%%%%%%%%%%%% 2.g - repeat NN experiments from Q1 with
%%%%%%%%%%%%%%%%%%%%%%%%% classification data %%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%% 1e %%%%%%%%%%%%%%%%%%%%%

adj_matrix = zeros(8,8);
adj_matrix(1,3:8) = ones(1,6);
adj_matrix(2,3:7) = ones(1,5);
adj_matrix(3:7,8) = ones(5,1);
xx = zeros(length(x_train),2);
xx(:,1) = 1;
xx(:,2) = x_train;

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

    learning_rate = 0.03/(2^i);
    [y_predict_2g_e, mse_over_time, neurons] = NeuralNetwork(adj_matrix, xx, y_train, 2, 5, 1, learning_rate, num_itrs);
    MSE_time(:,i) = mse_over_time;
    %plot(t,mse_over_time);
    
    MSE = mse_over_time(num_itrs);
    MSE_at_itrs(i) = MSE;
    %disp(strcat('MSE is::',num2str(MSE)));
    %disp(strcat('for learning_rate == ',num2str(learning_rate)));
    if (MSE < MSE_min)
        MSE_min = MSE;
        lr_best = learning_rate;
        neurons_best = neurons;
        itr_best = i;
    end
end

%I need a better way to do this:
%figure;
%t = log10(t);
%plot(t,MSE_time(:,1),t,MSE_time(:,2),t,MSE_time(:,3),t,MSE_time(:,4),t,MSE_time(:,5),...
%     t,MSE_time(:,6),t,MSE_time(:,7),t,MSE_time(:,8),t,MSE_time(:,9),t,MSE_time(:,10));

% Use the best model (by learning rate) to predict for test data:
xx = zeros(length(x_test),2);
xx(:,1) = 1;
xx(:,2) = x_test;
[y_predict_2g_e, MSE_2g_e] = NeuralNetworkEval(adj_matrix, xx, y_test, 2, 5, 1, neurons_best);
classErr_2g_e = nnz(round(y_predict_2g_e) - y_test)/length(y_test);
figure; plot(x_test, y_test, 'yo', x_test, y_predict_2g_e, 'r+');
title('Neural Net vanilla');
%disp(strcat('Test MSE is::',num2str(MSE_2g_e)));
%disp(strcat('for learning_rate == ',num2str(lr_best)));
figure; plot(x_test, y_test, 'yo', x_test, binaryThresholdVector(y_predict_2g_e,0.5), 'r+');
title('Neural Net vanilla - thresholded');

%%%%%%%%%%%%%%%%%%% 1f %%%%%%%%%%%%%%%%%%%%%%%%%
% Random restarts

adj_matrix = zeros(8,8);
adj_matrix(1,3:8) = ones(1,6);
adj_matrix(2,3:7) = ones(1,5);
adj_matrix(3:7,8) = ones(5,1);
xx = zeros(length(x_train),2);
xx(:,1) = 1;
xx(:,2) = x_train;

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
    [y_predict_2g_f, mse_over_time, neurons] = NeuralNetwork(adj_matrix, xx, y_train, 2, 5, 1, learning_rate, num_itrs);
    MSE_time(:,i) = mse_over_time;
    %plot(t,mse_over_time);
    
    MSE = mse_over_time(num_itrs);
    MSE_at_itrs(i) = MSE;
    %disp(strcat('MSE is::',num2str(MSE)));
    %disp(strcat('for learning_rate == ',num2str(learning_rate)));
    if (MSE < MSE_min)
        MSE_min = MSE;
        lr_best = learning_rate;
        neurons_best = neurons;
        itr_best = i;
    end
end

% Use the best model (by MSE) to predict for test data:
xx = zeros(length(x_test),2);
xx(:,1) = 1;
xx(:,2) = x_test;
[y_predict_2g_f, MSE_2g_f] = NeuralNetworkEval(adj_matrix, xx, y_test, 2, 5, 1, neurons_best);
figure; plot(x_test, y_test, 'yo', x_test, y_predict_2g_f, 'r+');
title('Neural Net, random restarts');
figure; plot(x_test, y_test, 'yo', x_test, binaryThresholdVector(y_predict_2g_f,0.5), 'r+');
classErr_2g_f = nnz(round(y_predict_2g_f) - y_test)/length(y_test);
title('Neural Net, random restarts - thresholded');
%disp(strcat('Test MSE is::',num2str(MSE_2g_f)));
%disp(strcat('for learning_rate == ',num2str(lr_best)));
figure; hist(MSE_at_itrs,10);
title('Neural Net, random restarts, MSE histogram');

%%%%%%%%%%%%%%%%%%%%%%% 1g %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Random restarts and up to 10 hidden logistical units
kmax = 10;
restarts = 20;

xx = zeros(length(x_val),2);
xx(:,1) = 1;
xx(:,2) = x_val;

num_itrs = 10000;
MSE_over_ks = zeros(restarts, kmax);
t = 1:num_itrs;
MSE_min = zeros(kmax,1)+Inf;
MSE_min_all = Inf;
k_best = 1;
itr_best = ones(kmax,1);
neurons_best = cell(kmax);
neurons_best_all = 0;



for k_2g_g=1:kmax %number of features
    for i=1:restarts %random restarts
    

        adj_matrix = zeros(3+k_2g_g,3+k_2g_g);
        adj_matrix(1,3:3+k_2g_g) = ones(1,k_2g_g+1);
        adj_matrix(2,3:2+k_2g_g) = ones(1,k_2g_g); %ones(1,k)
        adj_matrix(3:2+k_2g_g,3+k_2g_g) = ones(k_2g_g,1);

        learning_rate = 0.03/(2^3); %Best LR from before
        [y_predict_2g_g, mse_over_time, neurons] = NeuralNetwork(adj_matrix, xx, y_val, 2, k_2g_g, 1, learning_rate, num_itrs);

        MSE = mse_over_time(num_itrs);
        MSE_over_ks(i,k_2g_g) = MSE;
        %disp(strcat('MSE is::',num2str(MSE)));
        %disp(strcat('for k == ',num2str(k_2g_g)));
        if (MSE < MSE_min(k_2g_g))
            MSE_min(k_2g_g) = MSE;
            neurons_best{k_2g_g} = neurons;
            itr_best(k_2g_g) = i;
            if (MSE < MSE_min_all)
                k_best = k_2g_g;
                neurons_best_all = neurons;
            end
        end
    end
end

% Use the best model (by MSE) to predict for test data:
xx = zeros(length(x_test),2);
xx(:,1) = 1;
xx(:,2) = x_test;
k_2g_g = k_best;
adj_matrix = zeros(3+k_2g_g,3+k_2g_g);
adj_matrix(1,3:3+k_2g_g) = ones(1,k_2g_g+1);
adj_matrix(2,3:2+k_2g_g) = ones(1,k_2g_g); %ones(1,k)
adj_matrix(3:2+k_2g_g,8) = ones(k_2g_g,1);

[y_predict_2g_g, MSE_2g_g] = NeuralNetworkEval(adj_matrix, xx, y_test, 2, k_2g_g, 1, neurons_best_all);
classErr_2g_g = nnz(round(y_predict_2g_g) - y_test)/length(y_test);
figure; plot(x_test, y_test, 'yo', x_test, y_predict_2g_g, 'r+');
title('Neural Net, random restarts, k hidden logit units');
figure; plot(x_test, y_test, 'yo', x_test, binaryThresholdVector(y_predict_2g_g,0.5), 'r+');
title('Neural Net, random restarts, k hidden logit units - thresholded');
%disp(strcat('Test MSE is::',num2str(MSE_2g_g)));
%disp(strcat('for k == ',num2str(k_2g_g), '|| restart number == ',num2str(itr_best(k_2g_g))));

%%%%%%%%%%%%%%%%%%%%%%%% 1h %%%%%%%%%%%%%%%%%%%%%%%%%%%
% k = 10, and early stopping.
kmax = 10;
restarts = 1;

xx = zeros(length(x_val),2);
xx(:,1) = 1;
xx(:,2) = x_val;

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
[y_predict_2g_h, mse_over_time_val, neurons_all] = NeuralNetworkEarlyStop(adj_matrix, xx, y_val, 2, k, 1, learning_rate, num_itrs);

[MSE_val,min_index] = min(mse_over_time_val);
%disp(strcat('Min MSE index over validation set: ',num2str(min_index)));
% Use the index of the best MSE to choose the neurons for evaluating:
neurons_best_all = neurons_all{min_index};

% Use the best model (by MSE) to predict for test data:
xx = zeros(length(x_test),2);
xx(:,1) = 1;
xx(:,2) = x_test;
adj_matrix = zeros(3+k,3+k);
adj_matrix(1,3:3+k) = ones(1,k+1);
adj_matrix(2,3:2+k) = ones(1,k); %ones(1,k)
adj_matrix(3:2+k,8) = ones(k,1);

[y_predict_2g_h, MSE_2g_h] = NeuralNetworkEval(adj_matrix, xx, y_test, 2, k, 1, neurons_best_all);
classErr_2g_h = nnz(round(y_predict_2g_h) - y_test)/length(y_test);
figure; plot(x_test, y_test, 'yo', x_test, y_predict_2g_h, 'r+');
title('Neural Net, k = 10 hidden logits, early stopping');
figure; plot(x_test, y_test, 'yo', x_test, binaryThresholdVector(y_predict_2g_h,0.5), 'r+');
title('Neural Net, k = 10 hidden logits, early stopping - thresholded');
%disp(strcat('Test MSE is::',num2str(MSE)));

%%%%%%%%% results
disp(strcat('y_predict_2g_e: classErr -- ',num2str(classErr_2g_e),' -- MSE -- ',num2str(MSE_2g_e)));
disp(strcat('y_predict_2g_f: classErr -- ',num2str(classErr_2g_f),' -- MSE -- ',num2str(MSE_2g_f)));
disp(strcat('y_predict_2g_g: classErr -- ',num2str(classErr_2g_g),' -- MSE -- ',num2str(MSE_2g_g)));
disp(strcat('y_predict_2g_h: classErr -- ',num2str(classErr_2g_h),' -- MSE -- ',num2str(MSE_2g_h)));