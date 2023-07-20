This is a readme which contains everything related to this project.


PART 1 : DATASETS
The different datasets are in the folder "/CMF/data/timtey/datasets",
In this folder there are 4 different folders:
 - The first one "dataset" contains 3 files .txt for training, validation and test where we can find the ID of the subjects.
    and three other files .csv where we have the 57 labels per each ID (each line is a label).
 - "dataset4" contains the files which are used for the clustering project of tractography fibers:
   - "tractography_3.csv" contains all the tractography which are using for the training.
   - there are 3 tractography files one for each: training, validation, testing
   - then the files for the labeled fibers for the test(with a column with the number of cells for each bundle) the validation and the training the ones which are used are the files :
     - tracts_filtered_train_train_label_to_number_without_missing.csv
     - tracts_filtered_train_valid_label_to_number_without_missing.csv
     - tracts_filtered_train_test_label_to_number_nb_cells_without_missing_2_part.csv (because we take just 57 bundles for only one subject to get a quick and balance test)
In the csv files we have column x_min,x_max... these columns contain the bounds of the brain for each subject, it helps to normalize and get the fiber in the center of the icosahedron



PART 2 : DATAS
For the fibers who come from labeled bundles for each subject, the path is : "/CMF/data/timtey/tracts/archives"
each folder inside contains the 57 bundles of labeled fibers and for each bundle, there is one vtp file with features (EstimatedUncertainty,FA1,FA2,HemisphereLocataion,trace1,trace2)
and one file _DTI.vtk which contains other features (FA,MD,AD,RD) which are used when we train with the tractography fibers because we can just have these features for the tractography fibers.

For the different tractography they are in the folder : "/CMF/data/timtey/tractography/all"
In this folder you can find the brain_mask of every subject in vtk files.
But also for all the subjects that we were able to create a tractography, you can find for each subject 9 files, the biggest one _flip.vtp is the whole brain tractography without any features, then 4 files of this whole brain tractography _1.vtp(first part of the tractography), _2.vtp(second part), _3.vtp(third part), _4.vtp(fourth part) and finally we have these four parts which have the new features used for the model _1_DTI.vtp ...
PS: They are all in the same folder because it's easier when you want to load all of them for the training.

In the folders "/CMF/data/timtey/tractography/test", "/CMF/data/timtey/tractography/training", "/CMF/data/timtey/tractography/validation", it's where you have all the informations to create the tractography of each subject.

In the folder "/CMF/data/timtey/UKF" you can find a vtk file for each subject who contains all the 57 bundles.

