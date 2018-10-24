--------- Final Code including all parts -------------
---------------------------------------------------------------------

* Each code file has comments for the important parts.
* All the process are implemented on Matlab.

0. As explained earlier in the final report, set each sensor on the spot.
	- RSSI: After installing the program and putting the nodes(agent node, receiving/sending node) on the spots, connect the computer node to the computer.
		In this process, you need check which port you connect it to the computer and then, modify the varible representing this port number.
	- Seismic: set the seismic sensors such as upload the program(sketch arduino) into the arduino and connect amp and bluetooth module with it.
		   connect the bluetooth module to the computer and then, in the same as RSSI, check the port number and modify the variable. 

1. To use the remote communication, must check the port number for each of the sensor
	- go to the 'computer attribute' and, 'equipment operator', then check the port number for seismic sensors and RSSI computer node sensor.

2. To use the SVM algorithm, execute the make.m file first by pusing F5 

3. main.m is the one for you to commence the program. 
   - There are three main functions for RSSI, Acoustic, and Seismic.
   - each of them can be executed separately by choosing variables(RSSI_on, Acoustic_on, and Seismic_on) b/t false and true.

4. every data from each funcntion is saved in Data file.
   - RSSI: localization graph and raw data from node sensor(ubee430)
   - seismic: raw data from two seismic sensors and features used for classification 
   - acoustic: raw data from two acoustic sensors and features used for classification




# Explanation of each code file(.m)
- main.m: main file for implementing the program
- getFeature.m: each raw data from sesimic and acoustic sensors is processed to give features for classification
- getRSSI.m: Get RSSI data from node
- getSeismicData.m: Get seismic data 
- make.m : set the environments to use SVM algorithm
- openReceive.m & openSend.m : for the communication b/t Labview and matlab 
