function [ x , y ] = randgoal()

l1=8;				            % distance between frame '1' and '2'
l2=l1;					            % distance between frame '2' and '3'
l3=6;				            % distance between frame '3' and 'tool'
rmax=l1+l2+l3-1;						% maximum distance between (0,0) and tool frame
rmin=(l2^2+(l1-l3)^2)^0.5+1;		    % minimum distance between (0,0) and tool frame

% phi = (rand()*2-1)*pi;
phi = rand()*pi;

radius = ((rmax-rmin)/2.0)*((rand()*2)-1)+((rmax+rmin)/2.0);

x = abs(radius * cos(phi));
y = abs(radius * sin(phi));
% x = radius * cos(phi);
% y = radius * sin(phi);


