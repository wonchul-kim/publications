
----------------------------------------------------------
------ Feature extraction and Classification ------
----------------------------------------------------------

---- Prerequisites : read the file named 'libSVM_manual' to run 'main_SVM.m'
---- Feature extraction : main_feature.m
---- Classification validation : main_SVM.m
---- Raw data >>
	rawData_acoustic >	
		test1  ~ test25	: Quadrotor only
		test31 ~ test55	: Quadrotor+Jaguar
		test61 ~ test85	: Jaguar only
		test91 ~ test99	: 1 Person
		test101~ test109: 3 People
	rawData_seismic	 >	
		test1  ~ test25	: Quadrotor only
		test31 ~ test55	: Quadrotor+Jaguar
		test61 ~ test85	: Jaguar only
		test91 ~ test99	: 1 Person
		test101~ test109: 3 People

--- 1. main_feature.m : extract time/frequency domain feature data from raw data

	-- Parameters  >>
		save_feature = 1;   % 0: do not save result files, 1: save result files
		type = 2;           % 1: acoustic sensor data, 2: seismic sensor dat
		mic_num = 2;        % 1: sensor number 1, 2: sensor number 2
	
	-- Result data >>
		- result files are saved in the folder 'data_train'
		acoustic_1.m : feature data extracted from acoustic sensor number 1
		acoustic_2.m : feature data extracted from acoustic sensor number 2
		seismic_1.m  : feature data extracted from seismic sensor number 1
		seismic_2.m  : feature data extracted from seismic sensor number 2

--- 2. main_SVM.m : validate the feature data and save the last trained SVM prediction model

	-- Parameters  >>
		save_model = 0;     % 0: do not save trained model, 1: save trained model
		kernel_type = 2;    % 0: linear // 2: RBF
		data_fusion = 2;    % 0: acoustic only, 1: seismic only, 2: acoustic+seismic
		nr_valid = 100;     % Total number of validation processes

	-- Result model>>
		- result file named 'TRmodel.mat' is saved in the current folder
