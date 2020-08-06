function PCA
    function A = PCA(D, alpha)
        [n, d] = size(D);   %Data length and original dimentionality
        sigma = cov(D);     %Covariance Matrix
        [U, lambda] = eig(sigma);
        eigv = sortrows(table(diag(lambda), U), 1, 'descend');
        lambda = table2array(eigv(:,1));   %Eigen-Values
        U = table2array(eigv(:,2));     %Eigen-Vectors
        legalDims = zeros(size(d));
        ldEnd = 1;
        for r = 1:d
           f = sum(lambda(1:r)) / sum(lambda(1:d-1));
           if f >= alpha
               legalDims(ldEnd) = r;
               ldEnd = ldEnd + 1;
           end
        end
        legalDims = sort(legalDims);
        r = legalDims(1);   %Reduced Basis
        A = zeros([n, r]);
        for i = 1:n
           A(i, :) = U(1:r,:)*D(i, 1:d)';
        end
    end
    T = readtable('iris.txt');
    dataMatrix = table2array(T(:, 1:3));
    A = PCA(dataMatrix, 0.95);
    disp(A);
    scatter(A(:,1), A(:,2));
end
