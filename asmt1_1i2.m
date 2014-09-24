%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CONSOLIDATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Averaging all models:
y_predict_final_1 = y_predict_1a + y_predict_1b + y_predict_1c + y_predict_1d + ...
                  y_predict_1e + y_predict_1f + y_predict_1g + y_predict_1h;
y_predict_final_1 = y_predict_final_1 / 8;
MSE_final_1 = mean((y_predict_final_1-y_test_t).^2);
figure; plot(x_test_t, y_test_t, 'yo', x_test_t, y_predict_final_1, 'r+');

% Averaging top m models:
MSE_vector = [MSE_1a, MSE_1b, MSE_1c, MSE_1d, MSE_1e, MSE_1f, MSE_1g, MSE_1h];
y_predict_list = [y_predict_1a, y_predict_1b, y_predict_1c, y_predict_1d, ...
                  y_predict_1e, y_predict_1f, y_predict_1g, y_predict_1h];
[s, sorted_indices] = sort(MSE_vector);
m = 4;
y_predict_final_2 = zeros(length(y_predict_final_1),1);
for i=1:m
   y_predict_final_2 = y_predict_final_2 + y_predict_list(:,sorted_indices(i)); 
end
y_predict_final_2 = y_predict_final_2 / m;
MSE_final_2 = mean((y_predict_final_2-y_test_t).^2);
figure; plot(x_test_t, y_test_t, 'yo', x_test_t, y_predict_final_2, 'b+');

% Weighted average for all models:
% Let a be a cut-off factor for the minimum MSE so that 1/MSE+a doesn't
% blow up for very low MSE models:
a = 0.0000001;
MSE_weight_vector = [1/(MSE_1a+a), 1/(MSE_1b+a), 1/(MSE_1c+a), 1/(MSE_1d+a), ...
                     1/(MSE_1e+a), 1/(MSE_1f+a), 1/(MSE_1g+a), 1/(MSE_1h+a)];
MSE_weight_vector = MSE_weight_vector/sum(MSE_weight_vector); %normalize sum of weights to 1
y_predict_list = [y_predict_1a, y_predict_1b, y_predict_1c, y_predict_1d, ...
                  y_predict_1e, y_predict_1f, y_predict_1g, y_predict_1h];
m = 8;
y_predict_final_3 = zeros(length(y_predict_final_1),1);
for i=1:m
   y_predict_final_3 = y_predict_final_3 + MSE_weight_vector(i)*y_predict_list(:,i); 
end
MSE_final_3 = mean((y_predict_final_3-y_test_t).^2);
figure; plot(x_test_t, y_test_t, 'yo', x_test_t, y_predict_final_3, 'k+');

disp('MSEs for AVERAGING, AVERAGING TOP 4, MSE WEIGHTED AVERAGING:');
disp(num2str(MSE_final_1));
disp(num2str(MSE_final_2));
disp(num2str(MSE_final_3));