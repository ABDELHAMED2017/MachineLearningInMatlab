% Create and train an arbitrary architecture neural network
% where all transfer functions are sigmoidal except the output (linear).
% Structural assumptions: 
% - input structure corresponds to input layer,
% - nodes in output layer have no connections between them
%
% input - TxN, where T is the number of training cases, N is number of
%         features. Consequently, N == # of input nodes to this NN.
%         
%         example input structure:
%         input = zeros(100,2); %100 training cases, 2 input features
%         input(:,1) = 1; %first column all 1's
%         input(:,2) = x; %second column is given 1D data
%
% targets - Tx1, 1D vector of desired output values (TODO: extend code)
%
% adj_matrix - [a_ij] where 1 == connection from node i to node j, 0 o/w.
%              size is JxJ, where J == num neurons
% ex: create the following:
%
%                    x1       x2
%               x3       x4       x5 
%                        
%                        x6 
% 
% Where x1 --> x3,x4,x5,x6
%       x3 --> x3,x4,x5       
%       x3,x4,x5 --> x6
%
%   w = zeros(6,6); %w(x,y) defines connection(s) from x to y
%   w(1,3:6) = ones(1,4);
%   w(2,3:5) = ones(1,3);
%   w(3:5,6) = ones(3,1);
%
% learning_rate, num_itrs --> as expected
% num_input_nodes --> how many neurons are input nodes in your architecture
%                    (will check for consistency with input structure)
% num_hidden_nodes --> how many neurons are hidden nodes in your architecture
% num_output_nodes --> how many neurons are output nodes in your architecture
% y --> output, assumed to be 1D (similar to 'targets')
% leot --> log error over time

%%% PROBABILISTIC VERSION - logistic output node that gets thresholded.
% Log-likelihood of data reported. 

function [y, LL_time, neurons] = NeuralNetworkProb(adj_matrix, input, targets, num_input_nodes, num_hidden_nodes, num_output_nodes, ...
                           learning_rate, num_itrs)
                           
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
    
    % Initialize neuron weight matrix to random numbers b/w (0,1) where connections exist:
    neurons = 0.1*(rand(total_num_nodes,total_num_nodes)-0.5) .* (adj_matrix ~= 0);
      
    % Transfer function matrix
    g = zeros(num_training_cases, total_num_nodes);
    for i=1:num_inputs
       g(:,i) = input(:,i); 
    end
    
    % Error derivative structure:
    dedx = zeros(num_training_cases, total_num_nodes);
    
    % Store the log of error over time (iterations):
    LL_time = zeros(num_itrs,1);
    
    % Perform learning for num_itrs iterations:
    for itr=1:num_itrs
        
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
           g(:,j) = sigmoidVector(input_struct(:,j)); % now sigmoidal output
        end
        
        %%%%%%%%% BACKWARD PROPAGATION %%%%%%%%%%%%%
        
        % 1. Output Units:
        for j=jLast:total_num_nodes
            % Modify derivative form to be sigmoidal:
           dedx(:,j) = 2*(g(:,j) - targets) .* (g(:,j).*(1-g(:,j))); % Calculate error derivatives
           % Assumes a 1-D target vector, can be extended here 
        end
        
        % 2. Hidden Units: (looped over backwards for recursive computation)
        for m=(total_num_nodes-num_output_nodes):-1:(num_input_nodes+1)
            dedx(:,m) = dedx(:,m+1:total_num_nodes) * neurons(m,m+1:total_num_nodes)' .* (g(:,m).*(1-g(:,m)));
            % g*(1-g) is sigmoidal derivative
        end
        
        errordx = g' * dedx; % finish computing error derivatives via chain rule (see full derivation)
        
        %%%%%%%%%%%%% UPDATE %%%%%%%%%%%%%%%%
        
        neurons = neurons - learning_rate * errordx .* (adj_matrix ~= 0);
        
        LL_time(itr) = loglikelihood(targets,g(:,total_num_nodes));
        
    end
    
    
    y = g(:,total_num_nodes); %assuming output is only 1D
end

