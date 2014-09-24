% For one-dimensional data
function y = NearestNeighborK(x_vector, y_vector, length, x_test_case, k)
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
           for j=leftLimit:rightLimit
               y = y + y_vector(x_sorted_indices(j)); %corresponds to original indices in unsorted array
           end
           % at this point we've found our neighbors and we can quit this
           % loop to return. Don't forget to normalize:
           y = y / k;
           
           break;
        end
    end
    
    if (y==Inf) 
        %according to our algorithm, this means x_test_case is greater than any element
        %in the training vector, x_vector. Perform nearest neighbors at the
        %end:
        %disp(strcat('x_test_case larger than all others: ',num2str(x_test_case),'///',num2str(max(x_vector))));
        y = 0; %we're going to accumulate
        rightLimit = length;
        leftLimit = length - k;
        for j=leftLimit:rightLimit
            y = y + y_vector(x_sorted_indices(j)); %corresponds to original indices in unsorted array
        end
        y = y / (k+1); %adjust because we're at the end, it grabs an extra value
        
    end
    
    %disp(strcat('returning y: ',num2str(y)));
    
end