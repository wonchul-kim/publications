function [ total_reward,steps,Q1,Q2,Q3] = ArmEpisode( maxsteps, Q1, Q2, Q3 ,xf, yf , alpha, gamma,epsilon,statelist,actionlist,grafic, trace1, trace2,trace3, lambda )


global T1 T2 T3 xt yt TxtSteps grafica 

% initial state
T1 = 0;
T2 = 0;
T3 = 0;


steps = 0;
total_reward = 0;

% initial perception
% convert the continous state variables to an index of the statelist
s1   = DiscretizeState([(xf-xt) (yf-yt) ],statelist);
s2   = DiscretizeState([(xf-xt) (yf-yt) ],statelist);
s3   = DiscretizeState([(xf-xt) (yf-yt) ],statelist);

% selects an action using the epsilon greedy selection strategy
a1 = e_greedy_selection(Q1,s1,epsilon);
a2 = e_greedy_selection(Q2,s2,epsilon);
a3 = e_greedy_selection(Q3,s3,epsilon);

for i=1:maxsteps    
    
   
    
    % convert the index of the action into an action value
    action1 = actionlist(a1);    
    action2 = actionlist(a2);    
    action3 = actionlist(a3);    
    
    %do the selected action and get the next car state      				
    xp  = ArmDoAction([action1 action2 action3]);
       
    
    % observe the reward at state xp and the final state flag
    [r,f]        = ArmGetReward(xp,xf,yf);
    total_reward = total_reward + r;
    
    % convert the continous state variables in [xp] to an index of the statelist    
    sp1   = DiscretizeState([(xf-xt) (yf-yt) ],statelist);
    sp2   = DiscretizeState([(xf-xt) (yf-yt) ],statelist);
    sp3   = DiscretizeState([(xf-xt) (yf-yt) ],statelist);    
    
    % selects an action using the epsilon greedy selection strategy
    ap1 = e_greedy_selection(Q1,sp1,epsilon);
    ap2 = e_greedy_selection(Q2,sp2,epsilon);
    ap3 = e_greedy_selection(Q3,sp3,epsilon);
    
    
    % Update the Qtable, that is,  learn from the experience    
    [Q1, trace1] = UpdateSARSA_e( s1, a1, r, sp1, ap1, Q1, trace1, alpha, gamma, lambda);
    [Q2, trace2] = UpdateSARSA_e( s1, a1, r, sp1, ap1, Q2, trace2, alpha, gamma, lambda);
    [Q2, trace3] = UpdateSARSA_e( s1, a1, r, sp1, ap1, Q3, trace3, alpha, gamma, lambda);
    
    %update the current state
    s1 = sp1;
    s2 = sp2;
    s3 = sp3; 
    
    %update the current action
    a1 = ap1;
    a2 = ap2;
    a3 = ap3;
    grafic  = grafica; 
%     pxt(i)=xt;
%     pyt(i)=yt;
    if (grafic==true)
        set(TxtSteps,'string',strcat('Steps: ',int2str(steps+1)));
        setplot;
       
        %plot(xt,yt,'k')
        drawnow;
        %MOVIE(numel(MOVIE)+1) = getframe(gcf);
    end
    
    %increment the step counter.
    steps=steps+1;
    
    % if the car reachs the goal breaks the episode
    if (f==true)
        break
    end
   
end
%plot(pxt,pyt,'Color',[.7 .7 .7]);
%drawnow;

