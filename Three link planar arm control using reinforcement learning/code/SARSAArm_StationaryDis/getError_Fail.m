function [ out ] = getError_Fail( A, B)
% A: current_Q
% B: next_Q

scaler = size(A, 1) * size(A, 2);
error1 = abs(A-B);
error2 = sum(sum(error1));
error3 = error2/scaler;

out = error3;
end

