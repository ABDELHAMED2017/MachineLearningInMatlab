function E = MeanSquaredError(actual, prediction, length)

    E = 0;
    for i=1:length
       E = E + (actual(i) - prediction(i))^2; 
    end
    
    E = E/length;

end