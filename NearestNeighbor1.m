function [y, x_index] = NearestNeighbor1(x_vector, y_vector, length, x_test_case)

    %using L2 norm as distance function
    shortest_dist = (x_test_case-x_vector(1))^2;
    nearest_index = 1;
    
    for i=2:length
       dist_test = (x_test_case-x_vector(i))^2;
       if (dist_test < shortest_dist)
           shortest_dist = dist_test;
           nearest_index = i;
       end
    end

    y = y_vector(nearest_index);
    x_index = nearest_index;
end