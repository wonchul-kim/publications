function [ out ] = getError_Suc( A, B)
% A: current_Q
% B: next_Q

scaler = size(A, 1) * size(A, 2);
% A = A/norm(A);
% B = B/norm(B);
error1 = abs(A-B)/2;
error2 = sum(sum(error1));
error3 = error2/scaler;

out = error3;
end

