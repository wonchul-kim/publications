instrreset;             % reset the serial port information 
delete(instrfindall);   % make default for serial port information

% check the serial port number for each xbee
% and modify the number
com_1 = 'COM4';
com_2 = 'COM6';
s1 = serial(com_1,'Baudrate',9600);
s2 = serial(com_2, 'Baudrate', 9600); 
fopen(s1); fopen(s2);
disp('Com Port Open')

time = 0;           % initialize time variable
value1_Save = [];   % initialize seismic data matrix1
value2_Save = [];   % initialize seismic data matrix2
timeSave = [];      % initialize time matrix

disp('Staring to get data..........');
flushinput(s1); flushinput(s2); % delete previous data in the serial port
tic % to start the time
while toc < 4 % until 4s, if you want to collect the data more, increase it. unit is sec.
    while(s1.bytesavailable > 10 && s2.bytesavailable > 10)
            time = toc;
            value1 = fgetl(s1);             % get data from xbee1
            value2 = fgetl(s2);             % get data from xbee2
            value1_num = str2num(value1);   % traslate it into number
            value2_num = str2num(value2);   % traslate it into number

            disp('data : ');
            disp(value1_num);
            disp(value2_num);
            disp('--    --    --');
            disp('bytesavailable');
            disp(s1.bytesavailable);        % check how many data is received in serial port1
            disp(s2.bytesavailable);        % check how many data is received in serial port2
            disp('==============================================');
            value1_Save = [value1_Save, value1_num]; % save received data
            value2_Save = [value2_Save, value2_num]; % save received data
            timeSave = [timeSave, time];             % save time data
    end
end

disp('Starting to store the data............');
% make each time scale same
if (length(timeSave)~=length(value1_Save))||(length(timeSave)~=length(value2_Save))
    len = min([length(timeSave),length(value1_Save),length(value2_Save)]);
    timeSave = timeSave(1:len);
    value1_Save = value1_Save(1:len);
    value2_Save = value2_Save(1:len);
end
% make saved data into csv file
csvwrite('three1(9).csv', [timeSave', value1_Save']);
csvwrite('three2(9).csv', [timeSave', value2_Save']);

% plot
figure(); plot(timeSave, value1_Save, '-b');
figure(); plot(timeSave, value2_Save, '-b');

% close and clear the port
fclose(s1); fclose(s2);
delete(s1); delete(s2);
clear s1; clear s2;

