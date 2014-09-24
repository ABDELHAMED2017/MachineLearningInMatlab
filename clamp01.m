% For probability functions that need to have the log taken
function y = clamp01(x)
    y = zeros(size(x));
    len = length(x);
    for i=1:len
       if x(i) <= 0
           y(i) = 1e-10;
       elseif x(i) > 1
           y(i) = 1;
       else
           y(i) = x(i);
       end
    end
end