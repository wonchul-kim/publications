clear all; close all; clc;

%% Setting RSS data scan options
nr_node = 6;    % the number of target nodes
nr_avg = 5;     % the number of RSS scanning
port_number = 1;% the order of the serial port connected to the base node

%% Scan RSS data
rssi_offset = -45;

instrreset;
instrfind;
info = instrhwinfo('serial');
ComPort = info.SerialPorts{port_number};
s = serial(ComPort);
set(s, 'BaudRate', 115200);
set(s, 'Parity', 'none');
set(s, 'DataBits', 8);
set(s, 'StopBit', 1);
set(s, 'Timeout', 10);

fopen(s);
instrfind;

ref = ['7E';'FE'];
Packet = zeros(nr_node,1);
Pack_bool = zeros(nr_node,1);
flushinput(s);
count = 0;
while(1)
    count = count + 1;
    if mod(count,20) == 0
        flushinput(s);
    end
    packet_temp = fread(s, 15, 'uint8');
    packet_node = dec2hex(packet_temp);
    packet_tran = packet_node.';
    nodeID = packet_node(7,:);
    nodeID = hex2dec(nodeID);
    proto = (min(packet_node(1,:)==ref(1,:)) + min(packet_node(1,:)==ref(2,:)))>0 && (min(packet_node(15,:)==ref(1,:)) + min(packet_node(15,:)==ref(2,:)))>0;
    if proto && (nodeID>=1) && (nodeID<=6)
        rssi = [packet_node(11,:),packet_node(12,:)];
        if rssi(1)=='F'
            rssi_dec = -(hex2dec('FFFF')-hex2dec(rssi)+1)+rssi_offset;
        else
            rssi_dec = hex2dec(rssi)+rssi_offset;
        end
        if Pack_bool(nodeID,1) <= nr_avg
            Packet(nodeID,1) = Packet(nodeID,1) + rssi_dec;
            Pack_bool(nodeID,1) = Pack_bool(nodeID,1) + 1;
        end
        fprintf('Received packet  %c%c %c%c %c%c %c%c %c%c %c%c %c%c %c%c %c%c %c%c %c%c %c%c %c%c %c%c %c%c\n',packet_tran(:));
        fprintf('Received RSSI value  %f\n',rssi_dec);
    else
        fprintf('Bad packet : %c%c %c%c %c%c %c%c %c%c %c%c %c%c %c%c %c%c %c%c %c%c %c%c %c%c %c%c %c%c\n',packet_tran(:));
    end
    if sum(Pack_bool) == nr_avg*length(Pack_bool)
        break;
    end
end
fclose(s);
fprintf('Packet reception is done.');
Packet = round(Packet/nr_avg);
disp(Packet);
