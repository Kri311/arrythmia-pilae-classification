In this project never do on your own anything. I'll ask questions and suggestions
 you have to give me the appropriate answers and in this project I'll build this
  one without any help from you. and do not modify anything as we go on. I'll do
   it on my own. for context refer to the draft_paper.pdf in paper/

okay first let's take the dataset for our project,
From the MIT-BIH dataset
we are having 
.dat - which contains the raw ECG signals (just continuous signals)
.hea - contains all the metadatas (like sampling frequency, number of channels, signal format, gain, baseline, signal names, record length)
.atr - contains the clinically annoted beats in the ECG signals

for extracting proper data,
create a new directory named preprocessing/
add new python file named preprocessing.py
and in that python file 
first segment a fixed time window on the ECG samples around the R-peak from both ECG leads and we extract segment contains the local morphology of the heartbeat, including the P wave, QRS complex, and T wave, to the extent covered by the selected window. The corresponding segments from Lead 1 and Lead 2 are then aligned and combined to form a dual-lead representation of the same heartbeat.The corresponding segments from Lead 1 and Lead 2 are then aligned and combined to form a dual-lead representation of the same heartbeat.
if 198 samples are extracted from each lead, the two segments are concatenated to produce a 396-dimensional feature vector, allowing the model to simultaneously learn morphological information from both ECG leads.

now we got concatenated leads which has the proper morphological feature of the ECG signals

then we normalise it using per- beat Z score nomalisation in order to remove the noises exhibited by patients body size, electrodes placement, electrical noises,etc
and then apply global standard scaling across all  396 dimensions
then we export the data by stratified split of data into train & test 

the above part explains how we are going to preprocess the data in order to use it in our model. tell me if there is something we want to modify from the methodolofy and tell if there is any suggestions or things we need to add or remove or alter.

think carefully and implement it.... according to our available data