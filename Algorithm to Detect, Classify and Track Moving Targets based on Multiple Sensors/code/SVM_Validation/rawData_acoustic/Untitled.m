a = csvread('test3.csv');

plot(a(:, 1), '-k'); hold on;
legend('Multirotor');
xlabel('�ð� (time step)');
ylabel('���� (mV)');
xlim([0 10^5]);
