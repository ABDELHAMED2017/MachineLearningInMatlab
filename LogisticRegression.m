% Logistic regression
% 
% X - design matrix, T x N where every row is a training example in terms
%     of N features (the user needs to construct this before-hand)
%     ** assumes bias 1 included in the N features of input!
% y - training target outputs, T x 1
% learning_rate - as named.
% w_init_{min,max} - value range for which to initialize the weights as
%                    random numbers in between.
% plot_intermediate - plot the model as it is being fit to the data per
%                     iteration
%
% max_itrs and error_threshold are stopping/"convergence" criteria; the
% function will report which condition was met first. To nullify, set to
% Inf and -Inf respectively.
function [w, error_vector, LL, LL_time] = LogisticRegression(X, y, learning_rate, w_init_min, w_init_max, max_itrs, error_threshold, plot_intermediate)

    T = size(y,1); %number of training cases
    N = size(X,2); %number of features
    
    T_test = size(X,1); %should also be the same
    if T ~= T_test
       disp('Error: target vector and design matrix have incompatible dimensions');
       disp(strcat('Size of target vector: ',num2str(T)));
       disp(strcat('Number of training cases in design matrix: ',num2str(T_test)));
       w = Inf;
       training_error = Inf;
       return;
    end

    y_predict = zeros(T,1);
    error_vector = zeros(N,1);
    LL_time = zeros(max_itrs, 1);
    t = 1:T;
    
    % Initialize weights randomly within range:
    % Use the formula L + (U-L).*rand(100,100);
    % Where L is the lower bound, U is the upper bound, dimensions are
    % 100,100 or as desired.
    L = w_init_min;
    U = w_init_max;
    w = L + (U-L).*rand(N, 1);
    
    num_itrs = 0;
    LL = Inf; 
    
    %Original error:
    y_predict = sigmoidVector(X*w);
    error_vector = (1/T).* (X'*(y - y_predict));
    LL_prev = loglikelihood(y,y_predict);
    LL_original = LL_prev;
    w_prev = w;
    w_original = w;
    
    while(1)
        
        % 1. Predict an output:
        y_predict = sigmoidVector(X*w);
        if (max(y_predict) > 1)
            disp('yo');
        end
        
        % 2. Compute errors:
        error_vector = (1/T).* (X'*(y - y_predict));
        
        % 3. Update parameters by steepest descent:
        w = w + learning_rate * error_vector;
        
        if plot_intermediate == 1
           plot(t,y,t,y_predict); 
        end

        LL = loglikelihood(y,y_predict);
        
        if (LL < LL_prev) %you stepped in the wrong direction, try again - likelihood should increase
            L = L/2;
            U = U/2;
            w = L + (U-L).*rand(N, 1);
            learning_rate = learning_rate / 2;
        end
        LL_prev = LL;
        w_prev = w;
        
        num_itrs = num_itrs + 1;
        LL_time(num_itrs) = LL;
        %Stopping criteria:
        if num_itrs >= max_itrs
           break; 
        end
        % Not going to end on loglikelihood threshold, just use itrs
        %if LL <= error_threshold
        %   break 
        %end
    end

    disp('Logistic Regression Complete.');
    disp(strcat('Number of iterations:: ',num2str(num_itrs)));
    disp(strcat('Loglikelihood:: ',num2str(LL)));
end

function g = sigmoid(z)
    g = 1/(1+exp(-z));
end

function g = sigmoidVector(z)
    g = zeros(size(z));
    len = length(z);
    for i=1:len
       g(i) = sigmoid(z(i)); 
    end
end

