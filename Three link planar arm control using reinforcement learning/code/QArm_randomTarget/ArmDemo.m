function  ArmDemo( maxepisodes )

global TxtEpisode goal f2 grafica 

clc
statelist   = BuildStateList();  % the list of states
actionlist  = BuildActionList(); % the list of actions

nstates     = size(statelist,1);
nactions    = size(actionlist,1);

Q1           = BuildQTable( nstates,nactions );  % the QTable
Q2           = BuildQTable( nstates,nactions );  % the QTable
Q3           = BuildQTable( nstates,nactions );  % the QTable

% %%%%%%%%% no learning (stable system)
% x =load('Qtables.mat');
% Q1 = x.Q1;
% Q2 = x.Q2;
% Q3 = x.Q3;
% alpha = 0.0;
% epsilon     = 0.01;

maxsteps    = 400;  % maximum number of steps per episode
alpha       = 0.2;  % learning rate
gamma       = 0.9;  % discount factor
epsilon     = 0.01; % probability of a random action selection
grafica     = false; % indicates if display the graphical interface



for i=1:maxepisodes 
    

    [xf yf]= randgoal();  % activate the random location of the goal        
    
    set(goal,'xdata',xf,'ydata',yf);
    set(TxtEpisode,'string',strcat('Episode: ',int2str(i)));     
    [total_reward,steps,Q1,Q2,Q3 ] = ArmEpisode( maxsteps, Q1, Q2, Q3, xf , yf, alpha, gamma,epsilon,statelist,actionlist);    
    
    if (mod(i,20)==0)
        save Qtables.mat Q1 Q2 Q3;
    end
    
    disp(['Espisode: ',int2str(i),' steps: ',int2str(steps),' reward: ',num2str(total_reward),' eplison:',num2str(epsilon)])
    
    xpoints(i) = i-1;
    ypoints(i) = steps;

    subplot(f2);
    plot(xpoints,ypoints);
    xlabel('Episodes');
    ylabel('Steps');
    setplot;
    drawnow;
    epsilon = epsilon * 0.99;
    

    
end





