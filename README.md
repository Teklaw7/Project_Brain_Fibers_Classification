This is a readme which contains everything related to this project.


PART 1 : DATASETS
The different datasets are in the folder "/CMF/data/timtey/datasets",
In this folder there are 11 different files:
 - 4 files which are datasets for tractography: one which lists the different files of the tractography, one for the test, one for the training, one for the validation
   we have the informations about the adress of each tractography, the subject ID, the bounds of the tractography brains which help for the normalisation of the fibers
 - the other files are used for the labeled datas for training, validation and testing. For each of them there is one file with all the subjects (for respectively training, validation and test), and one file without the missing subjects because we weren't able to create the tractography for these subjects. Just for the test there is one file "tracts_filtered_train_test_label_to_number_nb_cells_without_missing_2_part.csv"  which is smaller and is used for quicker test.



PART 2 : DATAS
For the fibers who come from labeled bundles for each subject, the path is : "/CMF/data/timtey/tracts/archives"
inside each folder contains the 57 bundles of labeled fibers and for each bundle, there is one vtp file with features (EstimatedUncertainty,FA1,FA2,HemisphereLocataion,trace1,trace2)
and one file _DTI.vtk which contains other features (FA,MD,AD,RD) which are used when we train with the tractography fibers because we can just have these features for the tractography fibers.

For the different tractography files they are in the folder : "/CMF/data/timtey/tractography/all"
In this folder you can find the brain_mask of every subject in vtk files.
But also for all the subjects that we were able to create a tractography, you can find for each subject 5 files, the biggest one _flip.vtp is the whole brain tractography without any features, finally we have the four other parts which have the new features used for the model _1_DTI.vtp , _2_DTI.vtp, _3_DTI.vtp, _4_DTI.vtp.
PS: They are all in the same folder because it's easier when you want to load all of them for the training.

There is also a folder "/CMF/data/timtey/tractography/all/Test_tract_Slicer" where are the results of the bundles saved from tractography after clustering model, there is one when the lights were used in the loss function and one where we didn't use the lights for the loss function.
In the folders "/CMF/data/timtey/tractography/test", "/CMF/data/timtey/tractography/training", "/CMF/data/timtey/tractography/validation", it's where you have all the informations to create the tractography of each subject.

In the folder "/CMF/data/timtey/UKF" you can find a vtk file for each subject who contains all the 57 bundles.

In the folder "/CMF/data/timtey/Lights" you can find the different lights used for experiences and the trained model if you need it.

In the folder "/CMF/data/timtey/DTI" you can find all the informations .nrrd files that you can load with 3DSlicer related to the DTI of each subject.
Same for the DWI in the folder "/CMF/data/timtey/DWI".

In the folder "/CMF/data/timtey/RESULTS" you can find all the results that we got during the project about clustering the fibers. in the folder "results_contrastive_learning_062723" you can find the results we got by using a batchsize of 10 and the align + 0.1*uniformity loss function between the lights and the labeled fibers more also the result for the classification on all sphere with projections in 3D, these results are the best we got with this loss function by training just the labeled fibers with projections in 3D you can run the script "result_labeled_fibers_3D.py" which is in tools in this project.
In the folder "/RESULTS/results_contrastive_learning_062623" you can find the results we got by using a batchsize of 10 and the align + 0.1*uniformity loss function between the lights and the labeled fibers more also the result for the classification on all hypersphere with projections in 128D, these results are the best we got with this loss function by training just the labeled fibers with projections in 128D you can run the script "result_labeled_fibers_128D.py" which is in tools in this project.
In the folder "/RESULTS/results_contrastive_learning_063023" you can find the results we got for clustering with labeled fibers and tractography fibers, by using a batchsize of 16 (for each batch we have the first half which contains 8 labeled fibers and the second half which contains 8 tractography fibers), we used for the loss function :
(align + 0.1* uniformity between the lights and labeled datas + classification between the lights and labeled datas) + (align + 0.1*uniformity between two augmentations of tractography fibers). The results are on the all sphere in 128D.
In the folder "/RESULTS/results_contrastive_learning_071023" you can find the results we got for clustering with labeled fibers and tractography fibers, by using a batchsize of 16 (for each batch we have the first half which contains 8 labeled fibers and the second half which contains 8 tractography fibers), we used for the loss function :
(align + 0.1* uniformity between the lights and labeled datas + classification between the lights and labeled datas) + (align + 0.1*uniformity between two augmentations of tractography fibers). The results are on the all sphere in 3D.
In the folder "/RESULTS/results_contrastive_learning_071823" you can find the results we got for clustering with labeled fibers and tractography fibers, by using a batchsize of 16 (for each batch we have the first half which contains 8 labeled fibers and the second half which contains 8 tractography fibers), we used for the loss function :
(align + 0.1* uniformity between two augmentations of labeled fibers) + (align + 0.1*uniformity between two augmentations of tractography fibers). The results are on the all hypersphere in 128D. This is the last result we got and shows us that the lights are finally maybe not needed to train the model.

PART 3 : PROJECT FILES
In this project, in the folder tools, you can find all the different tools which were used during all the project.
In the folder Nets, you can find 3 different networks :  "brain_module_cnn.py" was used for classification, "brain_module_cnn_contrastive_labeled.py" was used for contrastive learning  to train the model to cluster labeled fibers when we tried SimClr method as a loss function, "brain_module_cnn_contrastive_tractography_labeled.py" was used to train the model to cluster the labeled fibers and the tractography fibers.
In the folder Data_Loaders, you can find 2 different files which contain the different datasets and dataloaders used for the project:
"data_module_classification_or_contrastive_labeled.py" was used for classification and for contrastive learning  to train the model to cluster labeled fibers when we tried SimClr method as a loss function.
"data_module_contrastive_tractography_labeled.py" was used to train the model to cluster the labeled fibers and the tractography fibers.

In the folder Transformations you can find a file with the different transformations which were applied during the project, as normalization, random rotation and random stretching.

The file "logger.py" contains two classes which are used when we want to visualize the images obtained with FlyByCNN, the class "BrainNetImageLogger" used for classification and the class "BrainNetImageLogger_contrastive_tractography_labeled" used for clustering labeled fibers and tractography fibers.

The main.py file trains the models with training validation and test.
The main_test.py file executes the test of a pretrained model. 