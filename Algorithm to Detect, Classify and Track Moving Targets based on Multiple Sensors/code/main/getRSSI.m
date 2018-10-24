function Packet = getRSSI(s, nr_node, filename )
%%% ������ ���κ����� RSSI �����͸� �ް� ��ǻ�ͷ� ������ ��ȭ�ϴ� �Լ�

rssi_offset = -45; % �������� ������ �� Ķ���극�̼��� ���� ��Ⱚ

ref = ['7E';'FE'];
Packet = zeros(nr_node,1); % ������ ���� ����
Pack_bool = zeros(nr_node,1); % �����Ͱ� �� ������ Ȯ��
flushinput(s); % �ǽð����� �����͸� �ޱ� ���� ���̴� �����ʹ� ����
count = 0;
Save = []; time = 0;
while(1)
    count = count + 1; % ī��Ʈ
    if mod(count,20) == 0 % ���� ���� ���� �ִ� �ڿ� �ִ� �����͸� ���� ����
        flushinput(s);
    end
    packet_temp = fread(s, 15, 'uint8'); % ���� �����͸� ����
    packet_node = dec2hex(packet_temp); % ������ Ÿ�� ��ȯ
    packet_tran = packet_node.'; % Ʈ��������
    nodeID = packet_node(7,:); % ��� ��ȣ ����
    nodeID = hex2dec(nodeID); % Ÿ�� ��ȯ
    proto = (min(packet_node(1,:)==ref(1,:)) + min(packet_node(1,:)==ref(2,:)))>0 && (min(packet_node(15,:)==ref(1,:)) + min(packet_node(15,:)==ref(2,:)))>0;
    if proto && (nodeID>=2) && (nodeID<=1+nr_node) % ��� ��ȣ 2�� ���� �Ͽ� ������ ��� ��ȣ���� ����
        rssi = [packet_node(11,:),packet_node(12,:)]; % ��Ŷ �� RSSI ���� ��Ÿ���� �κ��� ����
        if rssi(1)=='F' % ��Ŷ Ÿ�Կ� ���� �з��ϰ� ��ü�� Ÿ�Ժ�ȯ�Ͽ� �б� ���� 10������ ��ȯ
            rssi_dec = -(hex2dec('FFFF')-hex2dec(rssi)+1)+rssi_offset; 
        else
            rssi_dec = hex2dec(rssi)+rssi_offset;
        end
        Packet(nodeID,1) = rssi_dec; % RSSI ���� ���� ���� -> ���׷��ǿ� ����� �뵵
        Pack_bool(nodeID,1) = 1; % �� ��帶�� ��� �ϳ����� �޵��� ����
%         fprintf('Received packet  %c%c %c%c %c%c %c%c %c%c %c%c %c%c %c%c %c%c %c%c %c%c %c%c %c%c %c%c %c%c\n',packet_tran(:));
%         fprintf('Received RSSI value  %f\n',rssi_dec); % �������� RSSI ��
        disp(Packet(2:end, 1)') % RSSI ���� ��帶�� ���
        time = time + 1;
        Save(time, :) = [nodeID, rssi_dec]; % ����
   % else
    %    fprintf('Bad packet : %c%c %c%c %c%c %c%c %c%c %c%c %c%c %c%c %c%c %c%c %c%c %c%c %c%c %c%c %c%c\n',packet_tran(:));
    end
    if sum(Pack_bool) == nr_node % �� ��帶�� �ּ� �ϳ����� ��� �Դ��� Ȯ��
        break;
    end
%     disp(Packet');
end
xlswrite(filename, Packet(2:end, 1)); % ������ ����
fprintf('Packet reception is done.');
% disp();

end

