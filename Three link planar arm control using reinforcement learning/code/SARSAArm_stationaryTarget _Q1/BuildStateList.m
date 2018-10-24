function [ states ] = BuildStateList
%BuildStateList builds a state list from a state matrix

% x1 = [-1 -0.5 -0.25 0 0.25 0.5 1];%difference in x
% x2 = [-1 -0.5 -0.25 0 0.25 0.5 1]; %difference in y
x1 = -90:10:90;
x2 = -90:10:90;
x3 = -90:10:90;

I=size(x1,2);
J=size(x2,2);
K=size(x3, 2);
states=[];
index=1;
for i=1:I    
    for j=1:J
        for k = 1:K
            states(index,1)=x1(i);
            states(index,2)=x2(j);
            states(index,3) = x3(k);
            index=index+1;  
         end
    end
end
