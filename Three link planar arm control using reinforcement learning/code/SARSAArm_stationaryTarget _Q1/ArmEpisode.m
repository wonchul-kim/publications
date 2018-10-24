function [ total_reward,steps,Q1 ] = ArmEpisode( maxsteps, Q1,xf, yf , alpha, gamma,epsilon,statelist,actionlist, idx)
%MountainCarEpisode do one episode of the mountain car
% maxstepts: the maximum number of steps per episode
% Q: the current QTable
% alpha: the current learning rate
% gamma: the current discount factor
% epsilon: probablity of a random action
% statelist: the list of states
% actionlist: the list of actions

global T1 T2 T3 TxtSteps grafica xt yt

% initial state
T1 = 0;
T2 = 0;
T3 = 0;
% forkin;

steps = 0;
total_reward = 0;

% initial perception
% convert the continous state variables to an index of the statelist
s1   = DiscretizeState([T1 T2 T3],statelist);


% selects an action using the epsilon greedy selection strategy
a1 = e_greedy_selection(Q1,s1,epsilon);


for i=1:maxsteps    
    
   
    % convert the index of the action into an action value
    action1 = actionlist(a1, :);    
   
    
    %do the selected action and get the next car state      				
    xp  = ArmDoAction([action1]);
       
    
    % observe the reward at state xp and the final state flag
    [r,f]        = ArmGetReward(xp,xf,yf);
    total_reward = total_reward + r;
    
    % convert the continous state variables in [xp] to an index of the statelist    
    sp1   = DiscretizeState([T1 T2 T3],statelist);


    % selects an action using the epsilon greedy selection strategy
    ap1 = e_greedy_selection(Q1,sp1,epsilon);
   

    % Update the Qtable, that is,  learn from the experience    
    Q1 = UpdateSARSA( s1, a1, r, sp1, ap1, Q1 , alpha, gamma );
    
    %update the current state
    s1 = sp1;
   
    
    %update the current action
    a1 = ap1;
   
    grafic  = grafica; 
    pxt(i)= xt;
    pyt(i)= yt;
    if (grafic==true)
        set(TxtSteps,'string',strcat('Steps: ',int2str(steps+1)));
        setplot;
       
        plot(xt,yt,'k')
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
% 
% if rem(idx, 2000) == 0 
%     plot(pxt,pyt,'Color',[.7 .7 .7]);
%     drawnow;
% end

