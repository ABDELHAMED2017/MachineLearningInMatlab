%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CONSOLIDATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Averaging all models:
y_predict_final_1 = y_predict_2a + y_predict_2b + y_predict_2c + y_predict_2d + ...
                  y_predict_2e + y_predict_2f + y_predict_2g_e + y_predict_2g_f + ...
                  y_predict_2g_g + y_predict_2g_h + y_predict_2h_e + y_predict_2h_f + ...
                  y_predict_2h_g + y_predict_2h_h;
y_predict_final_1 = y_predict_final_1 / 14;
classErr_final_1 = nnz(binaryThresholdVector(y_predict_final_1,0.5) - y_test)/length(y_test);
figure; plot(x_test, y_test, 'yo', x_test, y_predict_final_1, 'r+');
title('Average of all models weighted equally');
figure; plot(x_test, y_test, 'yo', x_test, binaryThresholdVector(y_predict_final_1,0.5), 'r+');
title('Average of all models weighted equally - thresholded');

% Averaging top m models:
classErr_vector = [classErr_2a, classErr_2b, classErr_2c, classErr_2d, classErr_2e, classErr_2f, ...
                   classErr_2g_e, classErr_2g_f, classErr_2g_g, classErr_2g_h, ...
                   classErr_2h_e, classErr_2h_f, classErr_2h_g, classErr_2h_h];
y_predict_list = [y_predict_2a, y_predict_2b, y_predict_2c, y_predict_2d, ...
                  y_predict_2e, y_predict_2f, y_predict_2g_e, y_predict_2g_f, ...
                  y_predict_2g_g, y_predict_2g_h, y_predict_2h_e, y_predict_2h_f, ...
                  y_predict_2h_g, y_predict_2h_h];
[s, sorted_indices] = sort(classErr_vector);
m = 5;
y_predict_final_2 = zeros(length(y_predict_final_1),1);
for i=1:m %pick m lowerst classification error models
   y_predict_final_2 = y_predict_final_2 + y_predict_list(:,sorted_indices(i)); 
end
y_predict_final_2 = y_predict_final_2 / m;
classErr_final_2 = nnz(binaryThresholdVector(y_predict_final_2,0.5) - y_test)/length(y_test);
figure; plot(x_test, y_test, 'yo', x_test, y_predict_final_2, 'b+');
title('Average of top 5 models weighted equally');
figure; plot(x_test, y_test, 'yo', x_test, binaryThresholdVector(y_predict_final_2,0.5), 'b+');
title('Average of top 5 models weighted equally - thresholded');

% Weighted average for all models:
% Let a be a cut-off factor for the minimum classErr so that 1/classErr+a doesn't
% blow up for very low classErr models:
a = 0.001;
classErr_weight_vector = ones(size(classErr_vector));
classErr_weight_vector = classErr_weight_vector ./ (classErr_vector+a);
classErr_weight_vector = classErr_weight_vector/sum(classErr_weight_vector); %normalize sum of weights to 1

m = length(classErr_weight_vector);
y_predict_final_3 = zeros(length(y_predict_final_1),1);
for i=1:m
   y_predict_final_3 = y_predict_final_3 + classErr_weight_vector(i)*y_predict_list(:,i); 
end
classErr_final_3 = nnz(binaryThresholdVector(y_predict_final_3,0.5) - y_test)/length(y_test);
figure; plot(x_test, y_test, 'yo', x_test, y_predict_final_3, 'k+');
title('Average of all models weighted by classification error');
figure; plot(x_test, y_test, 'yo', x_test, binaryThresholdVector(y_predict_final_2,0.5), 'k+');
title('Average of all models weighted by classification error - thresholded');

disp('classErrs for AVERAGING, AVERAGING TOP 5, classErr WEIGHTED AVERAGING:');
disp(num2str(classErr_final_1));
disp(num2str(classErr_final_2));
disp(num2str(classErr_final_3));

figure;
plot(x_test,y_predict_2a,'*',x_test,y_predict_2b,'*',x_test,y_predict_2c,'*',x_test,y_predict_2d,'*',...
     x_test,y_predict_2e,'*',x_test,y_predict_2f,'*',x_test,y_predict_2g_e,'*',x_test,y_predict_2g_f,'*',...
     x_test,y_predict_2g_g,'*',x_test,y_predict_2g_h,'*',x_test,y_predict_2h_e,'*',x_test,y_predict_2h_f,'*',...
     x_test,y_predict_2h_g,'*',x_test,y_predict_2h_h,...
     x_train,y_train,'k+',x_test,y_test,'ko');