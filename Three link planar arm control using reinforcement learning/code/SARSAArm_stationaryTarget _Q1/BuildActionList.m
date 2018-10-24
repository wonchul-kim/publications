function [ actions ] = BuildActionList
%BuildActionList

%actions for robot arm problem
x1  = [-5 ; -2 ;-1 ; 0 ; 1 ; 2 ; 5];
x2  = [-5 ; -2 ;-1 ; 0 ; 1 ; 2 ; 5];
x3  = [-5 ; -2 ;-1 ; 0 ; 1 ; 2 ; 5];

I=size(x1,1);
J=size(x2,1);
K=size(x3,1);
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

actions = states;