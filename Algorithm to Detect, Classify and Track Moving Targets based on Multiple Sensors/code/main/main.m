close all; clear all; clc;


%%% ����ġ on/off : RSSI, ��������, ���⼾�� 
RSSI_on = true; com_0 = 'COM9';
Seismic_on = true; com_1 = 'COM8'; com_2 = 'COM7';
Acoustic_on = true;
instrreset; % ��Ʈ �ʱ�ȭ
RSSI_num = 0; % ������ ���� ��ȣ �ʱ�ȭ
% %%%% For RSSI %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if RSSI_on == true 
    %%%%%%% RSSI ���κ��� �����͸� ����
    s = serial(com_0); % �ø��� ��� ����
    set(s, 'BaudRate', 115200);
    set(s, 'Parity', 'none');
    set(s, 'DataBits', 8);
    set(s, 'StopBit', 1);
    set(s, 'Timeout', 10);
    fopen(s);
    disp('RSSI port openend.');
    
    %%%%%%%% localization�� ���� Gaussian Process Regression ���� %%%%%
    % ��� �� 
    row = 4; 
    col = 4;
    nr_node = row*col;
    diff = 1.4; % ��尣 ���� (m)
    
    % ��� ��ǥ ����
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

    % prediction_x : ��� ��ǥ�� �׸���ȭ��
    grid_size = 30;
    [x1,x2] = meshgrid(1:grid_size, 1:grid_size);
    prediction_x = [x1(:)'; x2(:)']./5;
    
    % ����þ� ���׷��� ������ ����
    sigma_n = 1;
    sigma_f = 1.1251;
    l = 0.7;

    % ����þȿ��� ���Ǵ� kernel �� ����
    error_function = @(x1, x2) sigma_n^2 * (sum(x1 == x2) == length(x1));
    kernel_function = @(x1, x2) sigma_f^2 * exp((x1-x2)' * (x1-x2) / (-2 * l^2));
    kernel = @(x1, x2) kernel_function(x1, x2) + error_function(x1, x2);

    % Kd (���׷��ǿ��� ���Ǵ� ���� �� �ϳ�)
    Kd = zeros(size(X_o, 2), size(X_o, 2)); 
    for i = 1:size(X_o,2)
        for j = i:size(X_o,2)
            Kd(i, j) = kernel(X_o(:, i), X_o(:, j));
        end
    end
    Kd = Kd + triu(Kd, 1)';

    % Kp (���׷��ǿ��� ���Ǵ� ���� �� �ϳ�)
    Kp = zeros(size(prediction_x, 2), size(prediction_x, 2));
    for i = 1:size(prediction_x, 2)
        for j = i:size(prediction_x, 2)
            Kp(i, j) = kernel_function(prediction_x(:, i), prediction_x(:, j));
        end
    end
    Kp = Kp + triu(Kp, 1)';

    % Kpd (���׷��ǿ��� ���Ǵ� ���� �� �ϳ�)
    Kpd = zeros(size(prediction_x, 2), size(X_o, 2));
    for i = 1:size(prediction_x, 2)
        for j = 1:size(X_o, 2)
            Kpd(i, j) = kernel_function(prediction_x(:, i), X_o(:, j));
        end
    end
end

%%%% For SESIMIC %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if Seismic_on == true
    Seismic_num = 0; % �ø��� ��� ����
    s1 = serial(com_1,'Baudrate',9600);
    s2 = serial(com_2, 'Baudrate', 9600);
    fopen(s1); fopen(s2);
    disp('Seismic port openend.');
end
%%%% For ACOUSTIC %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if Acoustic_on == true
    Acoustic_num = 0; % �ø��� ��� ����
    stat_lab = 0;
    w = openReceive('localhost', 5000); % Server to LabVIEW 
    r = openSend('0.0.0.0',5001);       % Listener
    disp('Acoustic port openend.');
end

%%%% For classification %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load('TRmodel.mat'); % �з��� �������� �ʱ�ȭ
feature_hist = [];
predict_result = [];
cnt = 1;

%%%% ITERATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
pauseTime = 1; % ������ �ð� ����

for i = 1:50
% while(1)
    if RSSI_on == true %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        disp('Getting RSSI >>>>>>>>>>>>>>>>>>>>');
        filenumber = RSSI_num; % ������ ��ȣ 1
        count = RSSI_num; % ������ ��ȣ 2
        filename = ['Data/RSSI/RSSI_data/data_', num2str(filenumber), '.xls']; % ���� �̸�
        RSSIData = getRSSI(s, nr_node, filename); % RSSI ������ �ޱ� ���� ����Լ� ����
        fprintf('operated RSSI Num: %d\n', RSSI_num);
        RSSI_num = RSSI_num + 1; % ���� ������ ��ȣ�� ������Ʈ
    end
    if Seismic_on == true %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        disp('Getting Seismic data >>>>>>>>>>>>>>>>>>>>>');
        SeismicData = getSeismicData(s1, s2, Seismic_num); % ���������κ��� �����͸� ����
        fprintf('operated SEISMIC Num: %d\n', Seismic_num);
        Seismic_num = Seismic_num + 1; % ���� ������ ��ȣ�� ������Ʈ
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
        Y_o = RSSIData(2:end)+100; % ���κ��� ���� RSSI ���� ����
        %%% GP�� ���� ���, �л갪 ���
        mu = Kpd / Kd * Y_o; 
        cov = 1.96 * sqrt(diag(Kp - Kpd/Kd * Kpd'));
        mu_grid = reshape(mu, grid_size, grid_size);
        
        % �׸��� ��ü�� RSSI���� �׷����� ���
        fig = figure();
        hold on;
        prediction_x_grid = reshape(prediction_x(1,:),grid_size,grid_size);
        prediction_y_grid = reshape(prediction_x(2,:),grid_size,grid_size);
        surf(prediction_x_grid, prediction_y_grid, mu_grid);
        xlabel('x'); ylabel('y'); zlabel('RSSI');
        
        % RSSI ���� �ִ��� ���� ã�Ƽ� ���
        coord = zeros(2,1);
        [temp1,temp2] = max(mu_grid); 
        [amplitude,coord(2)] = max(temp1);
        coord(1) = temp2(coord(2));
        coord = [prediction_x_grid(coord(2),coord(1)); prediction_y_grid(coord(2),coord(1))];
        disp('maximum value'); disp(amplitude);
        disp('coordinate'); disp(coord);
        set(stem3(coord(2), coord(1), amplitude, 'r.'), 'MarkerSize', 30);
        
        % �׷��� ����
        set(fig, 'PaperPositionMode', 'auto'); 
        print(fig, '-dpng', ['Data/RSSI/localization_graph/RSSI_', num2str(count), '.png']); 
        count = count + 1;
    end
    
    %%%%%%%%%% SVM calssification %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    disp('Classification operating >>>>>>>>>>>>>>>>>>>>>>>>>>>>>');
    test_data = getFeature(AcousticData, SeismicData); % Ư¡������ ���� ���ĸ� ��� �Լ��κ��� �����͸� ����
    test_label = ones(2,1);
    [predict_label,~,~] = svmpredict(test_label, test_data.', model); % �з��ϴ� �Լ� 
    %%% �з��� ������ ����
    feature_hist(:,:,cnt) = test_data;
    predict_result(cnt,:) = predict_label;
    
    disp('ALL DONE >>>>>>>>>>>>>>>>>>>>>>>>>>');
    cnt = cnt + 1;
    fprintf('\n\n Predicted target label = %d\n\n',predict_label);
    pause(pauseTime);
end
save feature_hist feature_hist;
save predict_result predict_result;
