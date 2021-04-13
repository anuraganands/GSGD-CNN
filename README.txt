Read Me for CNN-GSGD

Paper: A Strategic Weight Refinement Maneuver for Convolutional Neural Networks, IJCNN 2021 (TBP)
Programmer: Patrick Sharma
Supervisor: Dr. Anuraganand Sharma

The CNN-GSGD code has been written and tested with MATLAB-R2018a. 
Disclaimer: Ensure that you have the nnet toolbox installed under your MATLAB license.

Steps to setup CNN-GSGDon MATLAB-R2018a
You will find the following files in the installation folder:
	- GSGD_setup.bat
	- GSGD_remove.bat
	- 'code' folder
	- 'CNN_Main.m'
	
The 'code' folder consists od the 'CNN-GSGD' code, and all the other necessary code changes needed for ease of use - such as enabling/displaying GSGD(isGuided) from the neural network training options.

To setup GSGD on your MATLAB:
	1. Run the 'GSGD_setup.bat' file.
	2. Browse and Select the 'AAAI_CNN-GSGD' folder with the source code.	
	3. Select the destination 'nnet' folder in MATLAB toolboxes.
		e.g 'C:\Program Files\MATLAB\R2018a\toolbox\nnet'

	4. This will automatically rename the original 'cnn' folder to 'cnn_original' to keep the original matlab cnn files untouched.

	5. You may then add the following additional configurations in the network trainingOptions. 
	   Refer to the sample CNN_Main file

	---------------------------------------------------------------------------------------------------------------------
	|	Set 'isGuided' parameters to true to enalble GSGD and then supply 'Rho', 'RevisitBatchNum' and 					|
	|	'VerificationSetNum' values																						|
	|																													|
	|	Simply remove the above parameters to run without GSGD or set 'isGuided' to false								|
	|																													|
	|		'Rho' - number of iterations  to run for collection and checking of consistent data							|
	|				before guided approach is activated to update the weights with consistent							|
	|		        data																								|
	|																													|
	|		'RevisitBatchNum' - number of previous batches to revisit and 												|
	|							check how it performs on present batch weights											|	
	|																													|
	|		'VerificationSetNum' - number of batches to set aside at the beginning of each epoch. 						|
	|		                       Each batch gets picked randomly from this set to attain true							|
	|		                       error on weights updated by each batch during										|
	|		                       training																				|
	---------------------------------------------------------------------------------------------------------------------

	6. Setup training & Testing dataset and run the 'CNN_Main.m' file to begin training

Note: The actual CNN-GSGD code has been incorporated and can be viewed in the './toolbox/nnet/cnn/+nnet/+internal/Trainer.m' file from Lines 72 to 327.

To remove GSGD on your MATLAB:
	1. Run the 'GSGD_remove.bat' file.
	2. Select the 'nnet' folder in MATLAB toolboxes which contains the CNN-GSGD code.
		e.g 'C:\Program Files\MATLAB\R2018a\toolbox\nnet'

	3. This will automatically remove 'cnn' folder with GSGD, and restore the original 'cnn' folder back into matlab.
