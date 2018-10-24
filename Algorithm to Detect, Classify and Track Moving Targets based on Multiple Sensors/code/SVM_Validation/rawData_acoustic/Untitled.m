a = csvread('test3.csv');

plot(a(:, 1), '-k'); hold on;
legend('Multirotor');
xlabel('시간 (time step)');
ylabel('전압 (mV)');
xlim([0 10^5]);
