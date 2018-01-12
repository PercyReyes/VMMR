
%% obtain path to new dataset 

newSetPath = fullfile('New Dataset');

%% create image datastore of new dataset

new_dataset = imageDatastore(newSetPath);

[labelIdx,score] = predict(vmmr_classifier,new_dataset);

vmmr_classifier.Labels(labelIdx)
