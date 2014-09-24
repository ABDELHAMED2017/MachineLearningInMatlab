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

%y = NeuralNetworkProb(adj_matrix, input, targets, num_input_nodes, num_hidden_nodes, ...
%                   num_output_nodes, learning_rate, num_itrs)

% log errors over time:
num_itrs = 10000;
LL_time = zeros(num_itrs, 10);
t = 1:num_itrs;
LL_at_itrs = zeros(1,20);
LL_max = -Inf;
lr_best = 0;
itr_best = 1;
neurons_best = 0;
for i=3:3

    learning_rate = 0.03/(2^3);
    [y_predict_2h_e, LL_over_time, neurons] = NeuralNetworkProb(adj_matrix, xx, y_train, 2, 5, 1, learning_rate, num_itrs);
    LL_time(:,i) = LL_over_time;
    %plot(t,LL_over_time);
    
    LL = LL_over_time(num_itrs);
    LL_at_itrs(i) = LL;
    %disp(strcat('LL is::',num2str(LL)));
    %disp(strcat('for learning_rate == ',num2str(learning_rate)));
    if (LL > LL_max)
        LL_max = LL;
        lr_best = learning_rate;
        neurons_best = neurons;
        itr_best = i;
    end
end

%I need a better way to do this:
%figure;
%t = log10(t);
%plot(t,LL_time(:,1),t,LL_time(:,2),t,LL_time(:,3),t,LL_time(:,4),t,LL_time(:,5),...
%     t,LL_time(:,6),t,LL_time(:,7),t,LL_time(:,8),t,LL_time(:,9),t,LL_time(:,10));

% Use the best model (by learning rate) to predict for test data:
xx = zeros(length(x_test),2);
xx(:,1) = 1;
xx(:,2) = x_test;
[y_predict_2h_e, LL_2h_e] = NeuralNetworkEvalLL(adj_matrix, xx, y_test, 2, 5, 1, neurons_best);
classErr_2h_e = nnz(binaryThresholdVector(y_predict_2h_e,0.5) - y_test)/length(y_test);
figure; plot(x_test, y_test, 'yo', x_test, y_predict_2h_e, 'r+');
title('Neural Net vanilla');
%disp(strcat('Test LL is::',num2str(LL_2h_e)));
%disp(strcat('for learning_rate == ',num2str(lr_best)));
figure; plot(x_test, y_test, 'yo', x_test, binaryThresholdVector(y_predict_2h_e,0.5), 'r+');
title('Neural Net vanilla - thresholded');
clear LL_over_time;

%%%%%%%%%%%%%%%%%%% 1f %%%%%%%%%%%%%%%%%%%%%%%%%
% Random restarts

adj_matrix = zeros(8,8);
adj_matrix(1,3:8) = ones(1,6);
adj_matrix(2,3:7) = ones(1,5);
adj_matrix(3:7,8) = ones(5,1);
xx = zeros(length(x_train),2);
xx(:,1) = 1;
xx(:,2) = x_train;

%y = NeuralNetworkProb(adj_matrix, input, targets, num_input_nodes, num_hidden_nodes, ...
%                   num_output_nodes, learning_rate, num_itrs)

% log errors over time:
num_itrs = 10000;
LL_time = zeros(num_itrs, 10);
t = 1:num_itrs;
LL_at_itrs = zeros(1,20);
LL_max = -Inf;
lr_best = 0;
itr_best = 1;
neurons_best = 0;

for i=1:20

    learning_rate = 0.03/(2^3); %Best LR from before
    [y_predict_2h_f, LL_over_time, neurons] = NeuralNetworkProb(adj_matrix, xx, y_train, 2, 5, 1, learning_rate, num_itrs);
    LL_time(:,i) = LL_over_time;
    %plot(t,LL_over_time);
    
    LL = LL_over_time(num_itrs);
    LL_at_itrs(i) = LL;
    %disp(strcat('LL is::',num2str(LL)));
    %disp(strcat('for learning_rate == ',num2str(learning_rate)));
    if (LL > LL_max)
        LL_max = LL;
        lr_best = learning_rate;
        neurons_best = neurons;
        itr_best = i;
    end
end

% Use the best model (by LL) to predict for test data:
xx = zeros(length(x_test),2);
xx(:,1) = 1;
xx(:,2) = x_test;
[y_predict_2h_f, LL_2h_f] = NeuralNetworkEvalLL(adj_matrix, xx, y_test, 2, 5, 1, neurons_best);
figure; plot(x_test, y_test, 'yo', x_test, y_predict_2h_f, 'r+');
title('Neural Net, random restarts');
figure; plot(x_test, y_test, 'yo', x_test, binaryThresholdVector(y_predict_2h_f,0.5), 'r+');
classErr_2h_f = nnz(binaryThresholdVector(y_predict_2h_f,0.5) - y_test)/length(y_test);
title('Neural Net, random restarts - thresholded');
%disp(strcat('Test LL is::',num2str(LL_2h_f)));
%disp(strcat('for learning_rate == ',num2str(lr_best)));
%figure; hist(LL_at_itrs,10);
%title('Neural Net, random restarts, LL histogram');
clear LL_time;

%%%%%%%%%%%%%%%%%%%%%%% 1g %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Random restarts and up to 10 hidden logistical units
kmax = 10;
restarts = 20;

xx = zeros(length(x_val),2);
xx(:,1) = 1;
xx(:,2) = x_val;

num_itrs = 10000;
LL_over_ks = zeros(restarts, kmax);
t = 1:num_itrs;
LL_max = zeros(kmax,1)-Inf;
LL_max_all = -Inf;
k_best = 1;
itr_best = ones(kmax,1);
neurons_best = cell(kmax,1);
neurons_best_all = 0;



for k_2h_g=1:kmax %number of features
    for i=1:restarts %random restarts
    

        adj_matrix = zeros(3+k_2h_g,3+k_2h_g);
        adj_matrix(1,3:3+k_2h_g) = ones(1,k_2h_g+1);
        adj_matrix(2,3:2+k_2h_g) = ones(1,k_2h_g); %ones(1,k)
        adj_matrix(3:2+k_2h_g,3+k_2h_g) = ones(k_2h_g,1);

        learning_rate = 0.03/(2^3); %Best LR from before
        [y_predict_2h_g, LL_over_time, neurons] = NeuralNetworkProb(adj_matrix, xx, y_val, 2, k_2h_g, 1, learning_rate, num_itrs);

        LL = LL_over_time(num_itrs);
        LL_over_ks(i,k_2h_g) = LL;
        %disp(strcat('LL is::',num2str(LL)));
        %disp(strcat('for k == ',num2str(k_2h_g)));
        if (LL > LL_max(k_2h_g))
            LL_max(k_2h_g) = LL;
            neurons_best{k_2h_g} = neurons;
            itr_best(k_2h_g) = i;
            if (LL > LL_max_all)
                k_best = k_2h_g;
                neurons_best_all = neurons;
            end
        end
    end
end

% Use the best model (by LL) to predict for test data:
xx = zeros(length(x_test),2);
xx(:,1) = 1;
xx(:,2) = x_test;
k_2h_g = k_best;
adj_matrix = zeros(3+k_2h_g,3+k_2h_g);
adj_matrix(1,3:3+k_2h_g) = ones(1,k_2h_g+1);
adj_matrix(2,3:2+k_2h_g) = ones(1,k_2h_g); %ones(1,k)
adj_matrix(3:2+k_2h_g,8) = ones(k_2h_g,1);

[y_predict_2h_g, LL_2h_g] = NeuralNetworkEvalLL(adj_matrix, xx, y_test, 2, k_2h_g, 1, neurons_best_all);
classErr_2h_g = nnz(binaryThresholdVector(y_predict_2h_g,0.5) - y_test)/length(y_test);
figure; plot(x_test, y_test, 'yo', x_test, y_predict_2h_g, 'r+');
title('Neural Net, random restarts, k hidden logit units');
figure; plot(x_test, y_test, 'yo', x_test, binaryThresholdVector(y_predict_2h_g,0.5), 'r+');
title('Neural Net, random restarts, k hidden logit units - thresholded');
%disp(strcat('Test LL is::',num2str(LL_2h_g)));
%disp(strcat('for k == ',num2str(k_2h_g), '|| restart number == ',num2str(itr_best(k_2h_g))));
clear neurons_best_all;
clear neurons_best;
clear LL_max;

%%%%%%%%%%%%%%%%%%%%%%%% 1h %%%%%%%%%%%%%%%%%%%%%%%%%%%
% k = 10, and early stopping.
kmax = 10;
restarts = 1;

xx = zeros(length(x_val),2);
xx(:,1) = 1;
xx(:,2) = x_val;

num_itrs = 10000; 
LL_over_ks = zeros(restarts, kmax);
t = 1:num_itrs;
LL_max = zeros(kmax,1)-Inf;
LL_max_all = -Inf;
k_best = 1;
itr_best = ones(kmax,1);
neurons_best_all = 0;


k = 10;

adj_matrix = zeros(3+k,3+k);
adj_matrix(1,3:3+k) = ones(1,k+1);
adj_matrix(2,3:2+k) = ones(1,k); %ones(1,k)
adj_matrix(3:2+k,3+k) = ones(k,1);

learning_rate = 0.03/(2^3); %Best LR from before
[y_predict_2h_h, LL_over_time_val, neurons_all] = NeuralNetworkEarlyStopProb(adj_matrix, xx, y_val, 2, k, 1, learning_rate, num_itrs);

[LL_val,min_index] = min(LL_over_time_val);
%disp(strcat('Min LL index over validation set: ',num2str(min_index)));
% Use the index of the best LL to choose the neurons for evaluating:
neurons_best_all = neurons_all{min_index};

% Use the best model (by LL) to predict for test data:
xx = zeros(length(x_test),2);
xx(:,1) = 1;
xx(:,2) = x_test;
adj_matrix = zeros(3+k,3+k);
adj_matrix(1,3:3+k) = ones(1,k+1);
adj_matrix(2,3:2+k) = ones(1,k); %ones(1,k)
adj_matrix(3:2+k,8) = ones(k,1);

[y_predict_2h_h, LL_2h_h] = NeuralNetworkEvalLL(adj_matrix, xx, y_test, 2, k, 1, neurons_best_all);
classErr_2h_h = nnz(binaryThresholdVector(y_predict_2h_h,0.5) - y_test)/length(y_test);
figure; plot(x_test, y_test, 'yo', x_test, y_predict_2h_h, 'r+');
title('Neural Net, k = 10 hidden logits, early stopping');
figure; plot(x_test, y_test, 'yo', x_test, binaryThresholdVector(y_predict_2h_h,0.5), 'r+');
title('Neural Net, k = 10 hidden logits, early stopping - thresholded');
%disp(strcat('Test LL is::',num2str(LL)));
clear neurons_best_all;
clear LL_max;
%%%%%%%%% results
disp(strcat('y_predict_2h_e: classErr -- ',num2str(classErr_2h_e),' -- LL -- ',num2str(LL_2h_e)));
disp(strcat('y_predict_2h_f: classErr -- ',num2str(classErr_2h_f),' -- LL -- ',num2str(LL_2h_f)));
disp(strcat('y_predict_2h_g: classErr -- ',num2str(classErr_2h_g),' -- LL -- ',num2str(LL_2h_g)));
disp(strcat('y_predict_2h_h: classErr -- ',num2str(classErr_2h_h),' -- LL -- ',num2str(LL_2h_h)));