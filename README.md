# TDT4173-Project  
Repository for the machine learning project  
+ To run the code one should first extract the supplied 'Resources' folder into the 'TDT4173' <br>
(This ensures that the folder structure is maintained)

+ All runnable code is found in 'src/scripts', everything else is modules made for preprocessing/loading/etc.
These runnable files will do the following:<br>
-> Load the preprocessed data<br>
-> Perform hyperparameter search if necessary<br>
-> Train with the best hyperparameters<br>
-> Save the model and extract results<br>

+ All models and their results will be stored in the supplied 'Resources' folder <br>
(It should already contain everything as the experiments have already been ran)

+ We have mainly used PyCharm for development, which lets users auto import libraries from 'REQUIREMENTS.txt'.
If you are using a different IDE you might need to set the project working directory correctly before pathing will work.
We have added some code so that this might not be needed, but if you experience path errors check this first.

+ Hyperparameter search happens if no 'hyperparameter_search' folder is found in the specific model 'Resources/Models/...' path.
Since these should already exist the runnable codes will simply run the existing best hyperparameters.