function PCA

    function A = PCA(D, alpha)
        %[n, d] = size(D);
        sigma = cov(D);
        [U, lambda] = eig(sigma);
        lambda = sort(diag(lambda),"descend");
        disp(U);
        disp(lambda);
        disp(alpha);
        A = 5;
    end

    T = readtable('iris.txt');
    dataMatrix = table2array(T(:, 1:3));
    PCA(dataMatrix, 0.95);
end
