
----------------------------------------------------------
----------------- Code explanation ------------------
----------------------------------------------------------

--- This folder includes all the packages to run the algorithm.
--- All sub-folders include 'README.txt' file. Check the guidelines from each 'README.txt'.

-- 1. Codes for operating sensors and measuring multiple kinds of sensor data >>
	> 'Acoustic sensor' , 'RSS sensor' , Seismic sensor'
	- Acoustic sensor : package to run microphone
		             (LabVIEW program is required.)
	- RSS sensor : package to run RSS sensor
		     (TinyOS is required to construct hardware. Matlab is required to get data.)
	- Seismic sensor : package to run seismic sensor
		           (Arduino program is required to construct hardware. Matlab is required to get data.)

-- 2. Codes for extracting time/frequency domain features and running SVM classification algorithm
	> 'SVM_Validation'
	   (Matlab program is required. Additional installation of SVM library is required.)



-- 3. "main" folder is codes for integrating all the above algorithms and run the program.