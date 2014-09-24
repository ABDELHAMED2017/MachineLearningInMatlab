% For one-dimensional data
% For classification, where y_vector values must be 0 or 1
function y = NearestNeighborKprobability(x_vector, y_vector, length, x_test_case, k)
% precondition: k < length

    
    
    %Find the nearest k points in x_vector to x_test_case
    %Start by sorting x_vector:
    
    [x_sorted_vector, x_sorted_indices] = sort(x_vector); %lowest to highest
    y = Inf; %will be returned if we don't find neighbors
    
    for i=1:length
        if (x_test_case > x_sorted_vector(i))
            continue;
        else
           
           %found its spot in the ordering of x values. Now find the k nearest
           %neighbors, and return the mean of their y-values:
           numLeftNeighbors = ceil(k/2)-1; 
           %by convention, if odd # neighbors, take one more from the lower x values
           %note: this is completely arbitrary
           numRightNeighbors = floor(k/2); %because we're going to include index i as a neighbor
           %now store the neighbors:
           
           leftLimit = i-numLeftNeighbors;
           rightLimit = i+numRightNeighbors;
           
           if (i < leftLimit || leftLimit < 1)
               leftLimit = 1;
               rightLimit = k;
           end
           if (rightLimit > length)
               rightLimit = length;
               leftLimit = length - k;
           end
           %disp(leftLimit);
           %disp(rightLimit);
           
           y = 0; %we're going to accumulate
           numZeros = 0.1; %start with pseudocounts of 0.1
           numOnes = 0.1;
           for j=leftLimit:rightLimit
               jS = y_vector(x_sorted_indices(j));
               if jS==0
                   numZeros = numZeros + 1;
               elseif jS==1
                   numOnes = numOnes + 1;
               else
                   disp('Error: NearestNeighborKprobabilities received non-binary classification label, examine y_vector');
               end
           end
            % Set the output to probability of data == 1: 
           y = numOnes / (numZeros + numOnes);
           break;
        end
    end
    
    if (y==Inf) 
        %according to our algorithm, this means x_test_case is greater than any element
        %in the training vector, x_vector. Perform nearest neighbors at the
        %end:
        %disp(strcat('x_test_case larger than all others: ',num2str(x_test_case),'///',num2str(max(x_vector))));
        rightLimit = length;
        leftLimit = length - k;
        y = 0; %we're going to accumulate
        numZeros = 0.1; %start with pseudocounts of 0.1
        numOnes = 0.1;
        for j=leftLimit:rightLimit
            jS = y_vector(x_sorted_indices(j));
            if jS==0
                numZeros = numZeros + 1;
            elseif jS==1
                numOnes = numOnes + 1;
            else
                disp('Error: NearestNeighborKprobabilities received non-binary classification label, examine y_vector');
            end
        end
        % Set the output to probability of data == 1: 
        y = numOnes / (numZeros + numOnes);
    end
    
    %disp(strcat('returning y: ',num2str(y)));
    
end