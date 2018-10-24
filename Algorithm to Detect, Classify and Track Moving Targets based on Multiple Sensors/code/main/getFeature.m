function feature = getFeature(AcousticData, SeismicData)
%%% 주어진 raw 데이터를 통해 학습 가능한 특징점 추출 - weibull likelihood & STFT
nr_feature = 8; % 특징점 갯수 설정

feature = zeros(2*nr_feature,2);

for type = 1:2
    if type == 1
        Fs  = 25600;    % (Hz) Acoustic sensor sampling frequency
        X = AcousticData;
    elseif type == 2
        Fs  = 96;       % (Hz) Seismic sensor sampling frequency
        X = SeismicData;
    end
    
    for mic_num = 1:2 % 센서는 각각 두개이므로 하나씩 추출
        % Weibull likelihood **********************************************
        [M,~] = min(X);
        X(:,1) = X(:,1) + abs(M(1)) + 0.1;
        X(:,2) = X(:,2) + abs(M(2)) + 0.1;
        [parmhat,~] = wblfit(X(:,mic_num));
        nlogL = wbllike(parmhat,X(:,mic_num));
        
        % Short-time Fourier Transform ************************************
        wlen = 64;
        h = wlen/4;
        nfft = nr_feature*2-1;
        
        K = sum(hamming(wlen, 'periodic'))/wlen;
        [s,~,~] = spectrogram(X(:,mic_num),wlen,h,nfft,Fs);
        
        s = abs(s)/wlen/K;
        if rem(nfft, 2)                     % odd nfft excludes Nyquist point
            s(2:end, :) = s(2:end, :).*2;
        else                                % even nfft includes Nyquist point
            s(2:end-1, :) = s(2:end-1, :).*2;
        end
        
        s = 20*log10(s + 1e-6);
        
        if min(s) < 0
            s_min = min(s);
            if length(s_min) > 1
                s_min = min(min(s));
            end
            s = s + abs(s_min);
        end
        s = s/max(s);
        
        if type == 1
            feature(1,mic_num) = nlogL/1e5;
            feature(2:nr_feature,mic_num) = s(2:end);
        elseif type == 2
            feature(nr_feature+1,mic_num) = nlogL;
            feature(nr_feature+2:2*nr_feature,mic_num) = s(2:end);
        end
    end
end

end