function [y, LL] = NeuralNetworkEvalLL(adj_matrix, input, targets, num_input_nodes, num_hidden_nodes, num_output_nodes, neurons)
                           
    total_num_nodes = num_input_nodes + num_hidden_nodes + num_output_nodes;
    total_num_nodes_check1 = size(adj_matrix, 1);
    total_num_nodes_check2 = size(adj_matrix, 2);
    if (total_num_nodes ~= total_num_nodes_check1 || total_num_nodes ~= total_num_nodes_check2)
       disp('Error: the total number of nodes is not consistent:');
       disp('num_input_nodes,num_hidden_nodes,num_output_nodes:');
       disp(strcat(num2str(num_input_nodes),num2str(num_hidden_nodes),num2str(num_output_nodes)));
       disp('Adjacency Matrix dimensions:');
       disp(strcat(num2str(size(adj_matrix))));
       return;
    end
    
    num_inputs = size(input, 2);
    if (num_inputs ~= num_input_nodes)
       disp('Error: number of input nodes inconsistent with input structure:');
       disp('Number of input nodes:');
       disp(num2str(num_input_nodes));
       disp('Size of input structure:');
       disp(num2str(size(input)));
       return;
    end
    
    
    
    num_training_cases = size(input, 1);
    num_targets = size(targets, 1);
    if (num_targets ~= num_training_cases)
       disp('Error: target vector has incompatible size with input, should match num_training cases:');
       disp('input structure size: (TxN)');
       disp(num2str(size(input)));
       disp('target structure size: (Tx1)');
       disp(num2str(size(target)));
       return;
    end
    input_struct = input; %copy as the input structure will be modified 
      
    % Transfer function matrix
    g = zeros(num_training_cases, total_num_nodes);
    for i=1:num_inputs
       g(:,i) = input(:,i); 
    end
    
    
    % Compute the LL:
    LL = -Inf;

    %%%%%%%%%%% FORWARD PROPAGATION %%%%%%%%%%%

    % 1. Hidden Units:
    for j=(num_input_nodes+1):(num_input_nodes+num_hidden_nodes)
       input_struct(:,j) = g * neurons(:,j); %adds columns to input as needed 
       g(:,j) = 1./(1 + exp(-input_struct(:,j))); % Sigmoid transfer function
    end
    jLast = j+1; %stopped after last hidden unit, start at output units

    %2. Output Units:
    for j=jLast:total_num_nodes
       %disp(strcat('j:',num2str(j))); 
       input_struct(:,j) = g * neurons(:,j);
       g(:,j) = sigmoidVector(input_struct(:,j)); % Linear TF for output --> can be modified.
    end
    
    y = g(:,total_num_nodes); %assuming output is only 1D
    
    LL = loglikelihood(targets, y);
end