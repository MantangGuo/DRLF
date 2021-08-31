%
% Mattia Rossi (mattia.rossi@epfl.ch)
% Signal Processing Laboratory 4 (LTS4)
% Ecole Polytechnique Federale de Lausanne (Switzerland)
%
function B = blurmat(height, width, factor)
%
% Blur is defined as the convolution with a 2D box kernel.
% Since 2D box kernels are separable, convolution of a "height" x "width"
% image "X" with a 2D box kernel can be implemented by first filtering the
% columns of "X" with a 1D box kernel, and then its rows with the same kernel.
%
% Let "Bc" be the filtering matrix for the columns, and "Br" the one for the rows.
% Then "X" can be filtered as "Y = (Br*(Bc*X)')' = (Br * X' * Bc')' = Bc * X * Br' ".
%
% In vectorized form we have "Y(:) = kron(Br, Bc) * X(:)", hence "B = kron(Br, Bc)".
%
% Note that matrices "Bc" and "Br" may be different, despite implementing
% the same kernel, as image "X" may not be square.


% ==== Computes the blur matrix ===========================================

% Computes "Bc" and "Br".
Bc = avgmtx(height, factor);
Br = avgmtx(width, factor);

% Computes the blur matrix "B".
B = kron(Br, Bc);

end

function mtx = avgmtx(n, factor)

kernel = zeros((2 * (factor - 1)) + 1, 1);
kernel(factor:end) = 1 / factor;

diags = -(factor - 1):1:(factor - 1);

D = repmat(kernel', [n, 1]);
mtx = spdiags(D, diags, n, n);

end

