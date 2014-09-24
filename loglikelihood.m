function l = loglikelihood(targets,predictions)
    % note: added/subtracted 1e-20 epsilon fudge factor to avoid numerical
    % issues with taking logarithm of zero
    predictions = clamp01(predictions);
    l = targets'*log(predictions + 1e-10) + (ones(size(targets))-targets)'*(log(ones(size(targets))-predictions + 1e-10));
    if (isnan(l) == 1 || isreal(l) == 0)
        targets
        predictions
        pause(20);
    end
end