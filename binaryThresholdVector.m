function y = binaryThresholdVector(x, thresh)
    len = length(x);
    y = zeros(len,1);
    for i=1:len
       if x(i) > thresh
           y(i) = 1;
       else 
           y(i) = 0;
       end
    end
end