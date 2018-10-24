close all; clear all; clc;

E1 = load('Errors.mat');

S1 = load('Steps.mat');

I1 = load('Info.mat');


figure(1); 
plot(E1.xpoints, E1.ypoints, 'k');
xlabel('Episode'); ylabel('Error');


figure(2); 
plot(S1.xpoints, S1.yypoints, 'k');
xlabel('Episode'); ylabel('Step');

