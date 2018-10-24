function SeismicData = getSeismicData(s1, s2, dataNum)
%%% 진동센서로부터의 데이터를 무선으로 받는 함수
            
value1_Save = [];   % initialize seismic data matrix1
value2_Save = [];   % initialize seismic data matrix2
dataName = ['Data/Seismic/rawData/test', num2str(dataNum), '.xls'];

disp('Staring to get data..........');
flushinput(s1); flushinput(s2); % delete previous data in the serial port
tic % to start the time
idx = 1;
while toc < 4 % until 4s, if you want to collect the data more, increase it. unit is sec.
    while(s1.bytesavailable > 10 && s2.bytesavailable > 10)
            value1 = fgetl(s1);             % get data from xbee1
            value2 = fgetl(s2);             % get data from xbee2
            value1_num = str2num(value1);   % traslate it into number
            value2_num = str2num(value2);   % traslate it into number

            if isempty(value1_num) ~= true && isempty(value2_num) ~= true && size(value1_num, 1) == 1 && size(value2_num, 1) == 1

                value1_Save = [value1_Save, value1_num]; % save received data
                value2_Save = [value2_Save, value2_num]; % save received data
            end
    end
end
disp('Starting to store the data............');

value1_Save = value1_Save'; value2_Save = value2_Save'; % 트랜스포즈

if ((size(value1_Save, 1)) > (size(value2_Save, 1))) % 두 개의 센서의 길이를 동일하게 함
    value1_Save = value1_Save(1:size(value2_Save), :);
else
    value2_Save = value2_Save(1:size(value1_Save), :);
end

if isempty(value1_Save) ~= true && isempty(value2_Save) ~= true % 둘 중 하나의 센서에 값이 안들어올 경우
    xlswrite(dataName, [value1_Save, value2_Save]);
    SeismicData = [value1_Save, value2_Save];
else
    disp('Error: one or both of sensor does not work......!!!!!!!!!!!!!!!!1');
end

% plot
% figure(1); plot(timeSave, value1_Save, '-b');
% figure(2); plot(timeSave, value2_Save, '-b');
end


