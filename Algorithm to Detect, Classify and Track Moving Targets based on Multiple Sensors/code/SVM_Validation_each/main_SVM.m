clear all; close all; clc;

%% Load data
S1 = load('data_train\\acoustic_1.mat');
acoustic_1 = S1.feature;
S2 = load('data_train\\acoustic_2.mat');
acoustic_2 = S2.feature;
S3 = load('data_train\\seismic_1.mat');
seismic_1 = S3.feature;
S4 = load('data_train\\seismic_2.mat');
seismic_2 = S4.feature;

%% Validation settings
save_model = 0;  % 0: do not save trained model, 1: save trained model
kernel_type = 2; % 0: linear // 2: RBF
data_fusion = 2; % 0: acoustic only, 1: seismic only, 2: acoustic+seismic
nr_valid = 100;  % Total number of validation processes

%% Validate training data
num_data = 750;
num_total = length(acoustic_1)*2;
results = zeros(nr_valid,1);
for i = 1:nr_valid
    fprintf('\n********** Processing validation number %d. **********\n',i);
%     % Randomly select 500 validation data *********************************
%     test = randperm(num_total,500);
    
    % Generate matrix of full dataset and according label vector **********
    acoustic = [acoustic_1, acoustic_2];
    acoustic(1,:) = acoustic(1,:)/1e5;
    seismic = [seismic_1, seismic_2];
    label = [1*ones(num_data,1); 2*ones(num_data,1); 3*ones(num_data,1)];
    label = repmat(label,2,1);
    
    part = 3;
    temp_test = find(label==part);
    temp_len = length(temp_test);
    temp_index = randperm(temp_len,100);
    test = temp_test(temp_index);
    
    % Generate validation dataset *****************************************
    testA = acoustic(:,test);
    testS = seismic(:,test);
    if data_fusion == 2
        test_data = [testA; testS];
    elseif data_fusion == 0
        test_data = testA;
    elseif data_fusion == 1
        test_data = testS;
    end
    test_label = label(test);
    
    % Generate training dataset *******************************************
    trainA = acoustic; trainA(:,test) = [];
    trainS = seismic; trainS(:,test) = [];
    if data_fusion == 2
        train_data = [trainA; trainS];
    elseif data_fusion == 0
        train_data = trainA;
    elseif data_fusion == 1
        train_data = trainS;
    end
    train_label = label;
    train_label(test) = [];
    
    % Train SVM model and validate the model ******************************
    svm_type = '-t 2';
    model = svmtrain(train_label, train_data.', svm_type);
    [predict_label_L, accuracy_L, dec_values_L] = svmpredict(test_label, test_data.', model);
    
    % Save validation result history in terms of accuracy *****************
    results(i,1) = accuracy_L(1);
end
% Display mean accuracy ***************************************************
mean_accuracy = mean(results);
fprintf('\n The mean accuracy = %4.2f%%\n',mean_accuracy);

% Save trained SVM model **************************************************
if save_model
    save TRmodel model;
end
