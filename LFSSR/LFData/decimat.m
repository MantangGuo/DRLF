%
% Mattia Rossi (mattia.rossi@epfl.ch)
% Signal Processing Laboratory 4 (LTS4)
% Ecole Polytechnique Federale de Lausanne (Switzerland)
%
function [D, newHeight, newWidth] = decimat(height, width, factor)
%
% The decimation of an "height" x "width" image "X" can be carried out on
% the rows and the columns separately.
%
% Let "Dc" be the decimation matrix for the columns, and "Dr" the one for the rows.
% Then "X" can be decimated as "Y = (Dr*(Dc*X)')' = (Dr * X' * Dc')' = Dc * X * Dr' ".
%
% In vectorized form we have "Y(:) = kron(Dr, Dc) * X(:)", hence "D = kron(Dr, Dc)".


% ==== Computes the decimation matrix =====================================

% Computes "Dc" and "Dr".
Dc = decimtx1D(height, factor);
Dr = decimtx1D(width, factor);

% Computes the decimation matrix "D".
D = kron(Dr, Dc);
newHeight = size(Dc, 1);
newWidth= size(Dr, 1);

end

function mtx = decimtx1D(n, factor)

height = ceil(double(n) / factor);
width = n;

rows = zeros(height, 1);
cols = zeros(height, 1);

counter = 1;
for k = 1:1:height
    
    rows(k) = k;
    cols(k) = counter;
    
    counter = counter + factor;
    
end

mtx = sparse(rows, cols, ones(height, 1), height, width);

end

