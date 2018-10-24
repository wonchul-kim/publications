function Packet = getRSSI(s, nr_node, filename )
%%% 각각의 노드로부터의 RSSI 데이터를 받고 컴퓨터로 정보를 변화하는 함수

rssi_offset = -45; % 데이터의 노이즈 및 캘리브레이션을 위한 상쇄값

ref = ['7E';'FE'];
Packet = zeros(nr_node,1); % 데이터 저장 공간
Pack_bool = zeros(nr_node,1); % 데이터가 다 찻는지 확인
flushinput(s); % 실시간으로 데이터를 받기 위해 쌓이는 데이터는 삭제
count = 0;
Save = []; time = 0;
while(1)
    count = count + 1; % 카운트
    if mod(count,20) == 0 % 받지 못할 수도 있는 뒤에 있는 데이터를 위한 삭제
        flushinput(s);
    end
    packet_temp = fread(s, 15, 'uint8'); % 받은 데이터를 읽음
    packet_node = dec2hex(packet_temp); % 데이터 타입 변환
    packet_tran = packet_node.'; % 트랜스포즈
    nodeID = packet_node(7,:); % 노드 번호 저장
    nodeID = hex2dec(nodeID); % 타입 변환
    proto = (min(packet_node(1,:)==ref(1,:)) + min(packet_node(1,:)==ref(2,:)))>0 && (min(packet_node(15,:)==ref(1,:)) + min(packet_node(15,:)==ref(2,:)))>0;
    if proto && (nodeID>=2) && (nodeID<=1+nr_node) % 노드 번호 2번 부터 하여 마지막 노드 번호까지 받음
        rssi = [packet_node(11,:),packet_node(12,:)]; % 패킷 중 RSSI 값을 나타내는 부분을 저장
        if rssi(1)=='F' % 패킷 타입에 따라 분류하고 전체를 타입변환하여 읽기 편한 10진수로 변환
            rssi_dec = -(hex2dec('FFFF')-hex2dec(rssi)+1)+rssi_offset; 
        else
            rssi_dec = hex2dec(rssi)+rssi_offset;
        end
        Packet(nodeID,1) = rssi_dec; % RSSI 값은 따로 저장 -> 리그레션에 사용할 용도
        Pack_bool(nodeID,1) = 1; % 각 노드마다 적어도 하나씩은 받도록 설정
%         fprintf('Received packet  %c%c %c%c %c%c %c%c %c%c %c%c %c%c %c%c %c%c %c%c %c%c %c%c %c%c %c%c %c%c\n',packet_tran(:));
%         fprintf('Received RSSI value  %f\n',rssi_dec); % 오리지날 RSSI 값
        disp(Packet(2:end, 1)') % RSSI 값만 노드마다 출력
        time = time + 1;
        Save(time, :) = [nodeID, rssi_dec]; % 저장
   % else
    %    fprintf('Bad packet : %c%c %c%c %c%c %c%c %c%c %c%c %c%c %c%c %c%c %c%c %c%c %c%c %c%c %c%c %c%c\n',packet_tran(:));
    end
    if sum(Pack_bool) == nr_node % 각 노드마다 최소 하나씩은 들어 왔는지 확인
        break;
    end
%     disp(Packet');
end
xlswrite(filename, Packet(2:end, 1)); % 엑셀로 저장
fprintf('Packet reception is done.');
% disp();

end

