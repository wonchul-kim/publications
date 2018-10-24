clear all; close all; clc;

%% Feature extraction options
save_feature = 0;   % 0: do not save result files, 1: save result files
type = 2;           % 1: acoustic sensor data, 2: seismic sensor dat
mic_num = 1;        % 1: sensor number 1, 2: sensor number 2

%% Extract features (weibull likelihood, STFT)
if type == 1
    Fs  = 25600;    % (Hz) Acoustic sensor sampling frequency
    fprintf('***** Process acoustic data. Sensor number %d. *****\n',mic_num);
elseif type == 2
    Fs  = 96;       % (Hz) Seismic sensor sampling frequency
    fprintf('***** Process seismic data. Sensor number %d. *****\n',mic_num);
end
T = 1/Fs;

nr_feature = 8;
nr_window = 30;

data_index = 1:1:84;
data_index(56:60) = [];
data_index(26:30) = [];
nr_data = length(data_index);

feature = zeros(nr_feature,nr_data*nr_window);
for i = 1:25
    fprintf('Processing data number %d.\n',i);
    % Import raw data *****************************************************
    if type == 1
        filename = sprintf('rawData_acoustic\\test%d.csv',i);
    elseif type == 2
        filename = sprintf('rawData_seismic\\test%d.csv',i);
    end
    X_raw = load(filename);
    for j = 1:nr_window
        X_len = length(X_raw);
        window_step = round(X_len/100);
        X = X_raw((j-1)*window_step+1:X_len-window_step*(nr_window-j),:);
        
        % Weibull likelihood **********************************************
        [M,I] = min(X);
        X(:,1) = X(:,1) + abs(M(1)) + 0.1;
        X(:,2) = X(:,2) + abs(M(2)) + 0.1;
        [parmhat,parmci] = wblfit(X(:,mic_num));
        nlogL = wbllike(parmhat,X(:,mic_num));
        feature(1,(i-1)*nr_window+j) = nlogL;
        
        % Short-time Fourier Transform ************************************
        wlen = 64;
        h = wlen/4;
        nfft = nr_feature*2-1;
        
        K = sum(hamming(wlen, 'periodic'))/wlen;
        [s,f,t] = spectrogram(X(:,mic_num),wlen,h,nfft,Fs);
        
        s = abs(s)/wlen/K;
        if rem(nfft, 2)                     % odd nfft excludes Nyquist point
            s(2:end, :) = s(2:end, :).*2;
        else                                % even nfft includes Nyquist point
            s(2:end-1, :) = s(2:end-1, :).*2;
        end
        
        s = 20*log10(s + 1e-6);
        
        if min(s) < 0
            aa = abs(min(s));
            bb = [aa; aa; aa; aa; aa; aa; aa; aa];
            s = s + bb;
        end
        
        s = s/max(s);
        feature(2:end,(i-1)*nr_window+j) = s(2:end);
    end
end

figure()
plot(feature(7, 1:25)); 
% plot(feature(6, 31:55), '-b'); hold on
% plot(feature(6, 61:84), '-g'); hold on
% legend('Quadrotor', 'Qaudrotor+Jaguar', 'Jaguar');
legend('Multirotor');
xlabel('time(s)')
ylabel('STFT')


% %%
% % Delete empty matrices ***************************************************
% index = find(feature(1,:)==0);
% feature(:,index) = [];
% 
% % Save result feature matrix **********************************************
% fprintf('Saving.....\n');
% if save_feature
%     if type == 1
%         filename = sprintf('data_train\\acoustic_%d.mat',mic_num);
%     elseif type == 2
%         filename = sprintf('data_train\\seismic_%d.mat',mic_num);
%     end
% end
% save(filename,'feature');
% fprintf('Training data is generated.\n');

