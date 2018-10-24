close all; clear all; clc;

E1 = load('Errors_Q1.mat');
E3 = load('Errors_Q3.mat');

S1 = load('Steps_Q1.mat');
S3 = load('Steps_Q3.mat');

I1 = load('Info_Q1.mat');
I3 = load('Info_Q3.mat');


figure(1); subplot(2, 1, 1);
plot(E1.xpoints, E1.ypoints, 'r');
xlabel('Episode'); ylabel('Error');
legend('with a single Q');
subplot(2, 1, 2);
plot(E3.xpoints, E3.ypoints, 'k');
xlabel('Episode'); ylabel('Error');
legend('with multi Q');

figure(2); 
plot(S1.xpoints, S1.yypoints, 'r--', 'LineWidth', 1.5); hold on;
plot(S3.xpoints, S3.yypoints, 'k', 'LineWidth', 0.1);
xlabel('Episode'); ylabel('Step');
legend('with a single Q', 'with multi Q');

disp(I1.time)
disp(I3.time)
disp(I1.maxEpi)
disp(I3.maxEpi)
