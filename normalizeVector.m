% z = (x - mu) / sigma
% for all elements in x
function z = normalizeVector(x)
    z = (x - mean(x))/std(x);
end