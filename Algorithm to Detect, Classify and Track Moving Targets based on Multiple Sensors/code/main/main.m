close all; clear all; clc;


%%% 스위치 on/off : RSSI, 진동센서, 음향센서 
RSSI_on = true; com_0 = 'COM9';
Seismic_on = true; com_1 = 'COM8'; com_2 = 'COM7';
Acoustic_on = true;
instrreset; % 포트 초기화
RSSI_num = 0; % 데이터 저장 번호 초기화
% %%%% For RSSI %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if RSSI_on == true 
    %%%%%%% RSSI 노드로부터 데이터를 받음
    s = serial(com_0); % 시리얼 통신 셋팅
    set(s, 'BaudRate', 115200);
    set(s, 'Parity', 'none');
    set(s, 'DataBits', 8);
    set(s, 'StopBit', 1);
    set(s, 'Timeout', 10);
    fopen(s);
    disp('RSSI port openend.');
    
    %%%%%%%% localization을 위한 Gaussian Process Regression 셋팅 %%%%%
    % 노드 수 
    row = 4; 
    col = 4;
    nr_node = row*col;
    diff = 1.4; % 노드간 간격 (m)
    
    % 노드 좌표 설정
    X_o = [];
    idx = 1;
    for i = 0:row-1
        for j = 0:col-1
            X_o(idx, 1) = diff*j + 1;
            X_o(idx, 2) = (diff*(col-1) + 1) - diff*(i);
            idx = idx + 1;
        end
    end
    X_o = X_o';

    % prediction_x : 노드 좌표를 그리드화함
    grid_size = 30;
    [x1,x2] = meshgrid(1:grid_size, 1:grid_size);
    prediction_x = [x1(:)'; x2(:)']./5;
    
    % 가우시안 리그레션 변수값 설정
    sigma_n = 1;
    sigma_f = 1.1251;
    l = 0.7;

    % 가우시안에서 사용되는 kernel 식 정의
    error_function = @(x1, x2) sigma_n^2 * (sum(x1 == x2) == length(x1));
    kernel_function = @(x1, x2) sigma_f^2 * exp((x1-x2)' * (x1-x2) / (-2 * l^2));
    kernel = @(x1, x2) kernel_function(x1, x2) + error_function(x1, x2);

    % Kd (리그레션에서 사용되는 변수 중 하나)
    Kd = zeros(size(X_o, 2), size(X_o, 2)); 
    for i = 1:size(X_o,2)
        for j = i:size(X_o,2)
            Kd(i, j) = kernel(X_o(:, i), X_o(:, j));
        end
    end
    Kd = Kd + triu(Kd, 1)';

    % Kp (리그레션에서 사용되는 변수 중 하나)
    Kp = zeros(size(prediction_x, 2), size(prediction_x, 2));
    for i = 1:size(prediction_x, 2)
        for j = i:size(prediction_x, 2)
            Kp(i, j) = kernel_function(prediction_x(:, i), prediction_x(:, j));
        end
    end
    Kp = Kp + triu(Kp, 1)';

    % Kpd (리그레션에서 사용되는 변수 중 하나)
    Kpd = zeros(size(prediction_x, 2), size(X_o, 2));
    for i = 1:size(prediction_x, 2)
        for j = 1:size(X_o, 2)
            Kpd(i, j) = kernel_function(prediction_x(:, i), X_o(:, j));
        end
    end
end

%%%% For SESIMIC %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if Seismic_on == true
    Seismic_num = 0; % 시리얼 통신 설정
    s1 = serial(com_1,'Baudrate',9600);
    s2 = serial(com_2, 'Baudrate', 9600);
    fopen(s1); fopen(s2);
    disp('Seismic port openend.');
end
%%%% For ACOUSTIC %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if Acoustic_on == true
    Acoustic_num = 0; % 시리얼 통신 설정
    stat_lab = 0;
    w = openReceive('localhost', 5000); % Server to LabVIEW 
    r = openSend('0.0.0.0',5001);       % Listener
    disp('Acoustic port openend.');
end

%%%% For classification %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load('TRmodel.mat'); % 분류할 데이터의 초기화
feature_hist = [];
predict_result = [];
cnt = 1;

%%%% ITERATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
pauseTime = 1; % 루프간 시간 간격

for i = 1:50
% while(1)
    if RSSI_on == true %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        disp('Getting RSSI >>>>>>>>>>>>>>>>>>>>');
        filenumber = RSSI_num; % 데이터 번호 1
        count = RSSI_num; % 데이터 번호 2
        filename = ['Data/RSSI/RSSI_data/data_', num2str(filenumber), '.xls']; % 파일 이름
        RSSIData = getRSSI(s, nr_node, filename); % RSSI 정보를 받기 위한 재귀함수 설정
        fprintf('operated RSSI Num: %d\n', RSSI_num);
        RSSI_num = RSSI_num + 1; % 다음 데이터 번호로 업데이트
    end
    if Seismic_on == true %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        disp('Getting Seismic data >>>>>>>>>>>>>>>>>>>>>');
        SeismicData = getSeismicData(s1, s2, Seismic_num); % 진동센서로부터 데이터를 얻음
        fprintf('operated SEISMIC Num: %d\n', Seismic_num);
        Seismic_num = Seismic_num + 1; % 다음 데이터 번호로 업데이트
    end
    if Acoustic_on == true %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        disp('Getting Acoustic data >>>>>>>>>>>>>>>>>>>>>>>');
        while(1)
            if stat_lab == 1
                AcousticData = csvread('adata.csv');
                delete('adata.csv');
                dataName = ['Data/Acoustic/rawData/acoustic', num2str(Acoustic_num), '.mat'];
                save(dataName,'AcousticData');
%                 xlswrite(dataName, AcousticData);
                break;
            else
                command = 1;
                fwrite(w,command);
                stat_lab = fread(r,1);
                stat_lab = stat_lab - 48;
%                 disp(stat_lab);
            end
        end
        stat_lab = 0;
        fprintf('operated ACOUSTIC Num: %d\n', Acoustic_num);
        Acoustic_num = Acoustic_num + 1;
    end
    
    %%%%%%%%%% RSSI localization %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if RSSI_on == true        
        Y_o = RSSIData(2:end)+100; % 노드로부터 받은 RSSI 값을 저장
        %%% GP에 사용될 평균, 분산값 계산
        mu = Kpd / Kd * Y_o; 
        cov = 1.96 * sqrt(diag(Kp - Kpd/Kd * Kpd'));
        mu_grid = reshape(mu, grid_size, grid_size);
        
        % 그리드 전체의 RSSI값을 그래프로 출력
        fig = figure();
        hold on;
        prediction_x_grid = reshape(prediction_x(1,:),grid_size,grid_size);
        prediction_y_grid = reshape(prediction_x(2,:),grid_size,grid_size);
        surf(prediction_x_grid, prediction_y_grid, mu_grid);
        xlabel('x'); ylabel('y'); zlabel('RSSI');
        
        % RSSI 값이 최대인 지점 찾아서 출력
        coord = zeros(2,1);
        [temp1,temp2] = max(mu_grid); 
        [amplitude,coord(2)] = max(temp1);
        coord(1) = temp2(coord(2));
        coord = [prediction_x_grid(coord(2),coord(1)); prediction_y_grid(coord(2),coord(1))];
        disp('maximum value'); disp(amplitude);
        disp('coordinate'); disp(coord);
        set(stem3(coord(2), coord(1), amplitude, 'r.'), 'MarkerSize', 30);
        
        % 그래프 저장
        set(fig, 'PaperPositionMode', 'auto'); 
        print(fig, '-dpng', ['Data/RSSI/localization_graph/RSSI_', num2str(count), '.png']); 
        count = count + 1;
    end
    
    %%%%%%%%%% SVM calssification %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    disp('Classification operating >>>>>>>>>>>>>>>>>>>>>>>>>>>>>');
    test_data = getFeature(AcousticData, SeismicData); % 특징점으로 삼을 피쳐를 얻는 함수로부터 데이터를 얻음
    test_label = ones(2,1);
    [predict_label,~,~] = svmpredict(test_label, test_data.', model); % 분류하는 함수 
    %%% 분류한 데이터 저장
    feature_hist(:,:,cnt) = test_data;
    predict_result(cnt,:) = predict_label;
    
    disp('ALL DONE >>>>>>>>>>>>>>>>>>>>>>>>>>');
    cnt = cnt + 1;
    fprintf('\n\n Predicted target label = %d\n\n',predict_label);
    pause(pauseTime);
end
save feature_hist feature_hist;
save predict_result predict_result;
