% Algorithm and notation courtesy of: http://dl.dropbox.com/u/7412214/BackPropagation.pdf

function [L1weights, L2weights] = BackProp(inputFeatures, targetValues, epsilon, numItrs, numHiddenNeurons)

    % inputFeatures: every row is a feature vector, corresponding to an input with
    % which we will train our network. (numRows == number of training sets)
    % targetValues: target values / training data.
    % In general, targetValues can be a matrix of size (number of training sets,
    % numOutputNeurons) so that each output neuron can learn a value.     
    
    numInputNeurons = size(inputFeatures, 2); %i.e. the length of each row of inputFeatures which is size of each feature vec
    numOutputNeurons = size(targetValues, 2); 
    
    numTrainingSets = size(inputFeatures, 1); %the number of rows == number of training sets we have. 
    %targetValues must also have this exact same number of rows; assumed. 
     
    % Naming conventions: 
    % Input layer is I (i)
    % Hidden layer is J (j)
    % Output layer is K (k)
    
    % HiddenLayer and OutputLayer are sigmoid(inputs dot weights)
    % while the InputLayer is just rows of X_MAT (feature vectors)
    InputLayer = zeros(numInputNeurons, 1); %N.B.: this makes it a column
    HiddenLayer = zeros(numHiddenNeurons, 1); 
    OutputLayer = zeros(numOutputNeurons, 1); 
    
    % The weight matrices connecting the layers, and their energy
    % derivatives
    WeightMatrixIJ = rand(numInputNeurons, numHiddenNeurons) - 0.5; %randoms b/w (-0.5, 0.5)
    WeightMatrixJK = rand(numHiddenNeurons, numOutputNeurons) - 0.5; 
    EnergyWeightDwIJ = zeros(numInputNeurons, numHiddenNeurons); 
    EnergyWeightDwJK = zeros(numHiddenNeurons, numOutputNeurons);
    
    % A few helper vectors to organize and simplify lin alg operations
    OutputDelta = zeros(numOutputNeurons, 1); 
    HiddenDelta = zeros(numHiddenNeurons, 1); 
    currentTargets = zeros(numOutputNeurons, 1); %to compare with 
    
    % Let's do it. 
    for train=1:numItrs
        for iterator=1:numTrainingSets

            %%%%%%%% FORWARD PROPAGATION %%%%%%%%

            % Grab the inputs, which are rows of the inputFeatures matrix
            InputLayer = inputFeatures(iterator, :)'; %don't forget to turn into column 
            % Calculate the hidden layer outputs: 
            HiddenLayer = sigmoidVector(WeightMatrixIJ' * InputLayer); 
            % Now the output layer outputs:
            OutputLayer = sigmoidVector(WeightMatrixJK' * HiddenLayer);
            
%             %%%%%%% Debug stuff %%%%%%%% (for single valued output)
%             if (mod(train+iterator, 1000) == 0)
%                str = strcat('Output value: ', num2str(OutputLayer), ' | Test value: ', num2str(targetValues(iterator, :)')); 
%                disp(str);
%             end
            
            
            %%%%%%%% BACKWARDS PROPAGATION %%%%%%%%

            % Propagate backwards for the hidden-output weights
            currentTargets = targetValues(iterator, :)'; %strip off the row, make it a column for easy subtraction
            OutputDelta = (OutputLayer - currentTargets) .* OutputLayer .* (1 - OutputLayer); 
            EnergyWeightDwJK = HiddenLayer * OutputDelta'; %outer product
            % Update this layer's weight matrix:
            WeightMatrixJK = WeightMatrixJK - epsilon*EnergyWeightDwJK; %does it element by element

            % Propagate backwards for the input-hidden weights
            HiddenDelta = HiddenLayer .* (1 - HiddenLayer) .* WeightMatrixJK*OutputDelta; 
            EnergyWeightDwIJ = InputLayer * HiddenDelta'; 
            WeightMatrixIJ = WeightMatrixIJ - epsilon*EnergyWeightDwIJ; 

            
            
        end
        
        
        
    end
    
    % Return the weight matrices so the BackPropCalc function can compute
    % values in real-time post-training. 
    L1weights = WeightMatrixIJ; 
    L2weights = WeightMatrixJK; 
    
end



%element by element
function y = sigmoidVector(x)
    len = length(x);
    y = zeros(size(x,1),size(x,2)); 
    for i=1:len
        y(i) = sigmoid(x(i));
    end
end

function y = sigmoid(x)
    y = 1 / (1 + exp(-x)); 
end