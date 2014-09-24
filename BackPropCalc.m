function y = BackPropCalc(inputFeatureVec, L1weights, L2weights)

        %%% Note: can trade matrix transpose for vector transpose
        %%% and end up with a row at the return value. 

        % Calculate the hidden layer outputs: 
        HiddenLayer = sigmoidVector(L1weights' * inputFeatureVec); 
        % Now the output layer outputs:
        OutputLayer = SigmoidVector(L2weights' * HiddenLayer);
        
        y = OutputLayer; %vector or single number representing the score

end