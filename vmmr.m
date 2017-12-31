%% build the path to the training set  & test set.

trndbPath = fullfile('Coursework Database', 'Training Set');
testdbPath = fullfile('Coursework Database', 'Test Set');

%% Load training and test data uisng |iamgeDataStore|.

dbtrainingSet = imageDatastore(trndbPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
dbtestSet = imageDatastore(testdbPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

%% get the size of training set and test set to loop through both sets

trnSize = numel(dbtrainingSet.Files);
tstSize = numel(dbtestSet.Files);

%%---------- For training set ----------------- %
svm_trainingSet = struct('images',cell(1), 'keypoints', cell(1));
svm_testSet = struct('images', cell(1), 'keypoints', cell(1));

for i = 1:trnSize
    
    % read in every image from the training dataset
    I = readimage(dbtrainingSet, i);
    
    % convert image to grayscale
    I = rgb2gray(I);
    
    % resize image to 99x264
    I = imresize(I, [99 264]);
    
    % store processed images in training set for the 
    svm_trainingSet(i).images = I;
    
    % detect SURF features within each image
    svm_trainingSet(i).keypoints = detectSURFFeatures(I, 'MetricThreshold', 2000);
   
    %svm_trainingSet(i).keypoints = svm_trainingSet(i).keypoints.selectStrongest(50);
    
    % extract SURF features from image
    training_features(i,:) = sum(extractFeatures(svm_trainingSet(i).images, svm_trainingSet(i).keypoints, 'Upright', true, 'FeatureSize', 128));
    
end


%% ------------------ For test set ------------------ %

for i = 1:tstSize
    
    % read in every image from the training dataset
    I = readimage(dbtestSet, i);
    
    % convert image to grayscale
    I = rgb2gray(I);
    
    % resize image to 99x264
    I = imresize(I, [99 264]);
    
    % store processed images in training set for the 
    svm_testSet(i).images = I;
    
    % detect SURF features within each image
    svm_testSet(i).keypoints = detectSURFFeatures(I, 'MetricThreshold', 2000);

    %svm_testSet(i).keypoints = svm_testSet(i).keypoints.selectStrongest(50);
    
    % extract SURF features from image
    test_features(i,:) = sum(extractFeatures(svm_testSet(i).images, svm_testSet(i).keypoints, 'Upright', true, 'FeatureSize', 128));
  
end


%% obtain the training labels

trainingLabels = dbtrainingSet.Labels;
testLabels = dbtestSet.Labels;

%% -------- Train the svm classifier ---------------%
svm_model = fitcecoc(training_features, trainingLabels);

% cross validate classifier 
cv_svm_model = crossval(svm_model);


%% make predictions based on test images 
[predictedCars, score] = predict(svm_model, test_features);

%% display the results

% use confusiom matrix to evalutate the results
confMatrix = confusionmat(testLabels, predictedCars);
