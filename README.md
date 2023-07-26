This is a readme which contains everything related to this project.


PART 1 : DATASETS
The different datasets are in the folder "/CMF/data/timtey/datasets",
In this folder there are 11 different files:
 - 4 files which are datasets for tractography: one which lists the different files of the tractography, one for the test, one the training, one for the validation
   we have the informations about the adress of each tractography, the subject ID, the bounds of the tractography brains which help for the normalisation of the fibers
 - the other files are used for the labeled datas for training, validation and testing. For each them there is one file with all the subjects (for respectively training, validation and test), and one file without the missing subjects because we weren't able to create the tractography. Just for the test there is one file "tracts_filtered_train_test_label_to_number_nb_cells_without_missing_2_part.csv"  which is smaller and is used for quicker test.



PART 2 : DATAS
For the fibers who come from labeled bundles for each subject, the path is : "/CMF/data/timtey/tracts/archives"
each folder inside contains the 57 bundles of labeled fibers and for each bundle, there is one vtp file with features (EstimatedUncertainty,FA1,FA2,HemisphereLocataion,trace1,trace2)
and one file _DTI.vtk which contains other features (FA,MD,AD,RD) which are used when we train with the tractography fibers because we can just have these features for the tractography fibers.

For the different tractography they are in the folder : "/CMF/data/timtey/tractography/all"
In this folder you can find the brain_mask of every subject in vtk files.
But also for all the subjects that we were able to create a tractography, you can find for each subject 5 files, the biggest one _flip.vtp is the whole brain tractography without any features, finally we have the four other parts which have the new features used for the model _1_DTI.vtp , _2_DTI.vtp, _3_DTI.vtp, _4_DTI.vtp.
PS: They are all in the same folder because it's easier when you want to load all of them for the training.

There is also a folder "/CMF/data/timtey/tractography/all/Test_tract_Slicer" where are the results of the bundles saved from tractography after clustering model, there is one when the lights were used in the loss function and one where we didn't use the lights for the loss function.
In the folders "/CMF/data/timtey/tractography/test", "/CMF/data/timtey/tractography/training", "/CMF/data/timtey/tractography/validation", it's where you have all the informations to create the tractography of each subject.

In the folder "/CMF/data/timtey/UKF" you can find a vtk file for each subject who contains all the 57 bundles.

