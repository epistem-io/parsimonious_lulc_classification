import ee
import pandas as pd
# A single random split 
#extract pixel value for the labeled region of interest and partitioned them into training and testing data
#This can be used if the training/reference data is balanced across class and required more fast result
def extract_pixel_value(image, roi, class_property, scale=10, split_ratio = 0.5, tile_scale = 16):
     """
     extract the pixels value for supervised classification based on pre defined regions of interest (ROI) and
     partitioned the data into training and testing
     Parameters:
        image (ee.Image)
        Region of interest/ROI (feature collection)
        class_property/class_label
        scale(int), based on spatial resolution of the data
     Returns:
        tuple: (training_samples, testing_samples)
     """
     #create a random column
     roi_random = roi.randomColumn()
     #partioned the original training data
     training = roi_random.filter(ee.Filter.lt('random', split_ratio))
     testing = roi_random.filter(ee.Filter.gte('random', split_ratio))
     #extract the pixel values
     training_pixels = image.sampleRegions(
                        collection=training,
                        properties = [class_property],
                        scale = scale,
                        tileScale = tile_scale 
     )
     testing_pixels = image.sampleRegions(
                        collection=testing,
                        properties = [class_property],
                        scale = scale,
                        tileScale = tile_scale 
     )
     return training_pixels, testing_pixels


# the strafied kfold cross validation split for more robust partitioning between training and validation data.
# Ideal for imbalance dataset. 
def stratified_kfold(samples, class_property, k=5, seed=0):
    """
    Perform stratified kfold cross-validation split on input reference data.
    
    Parameters:
        samples (ee.FeatureCollection): training data or reference data which contain unique class label ID
        class_property (str): column name contain unique label ID
        k (int): Number of folds.
        seed (int): Random seed for reproducibility.
    
    Returns:
        ee.FeatureCollection: A collection of k folds. Each fold is a Feature
                              with 'training' and 'validation' FeatureCollections.
    """
    #define the logic for k-fold. It tells us how wide the split will be
    step = 1.0 / k
    #Threshold are similar to split ratio, in this context, an evenly space of data numbers. The results is a cut points for the folds,
    #in which each fold will takes sample whose asigned random number within the ranges
    thresholds = ee.List.sequence(0, 1 - step, step)
    #This code will aggregate unique class label into one distinct label
    classes = samples.aggregate_array(class_property).distinct()
    #function for create the folds using the given threshold
    def make_fold(threshold):
        threshold = ee.Number(threshold)
        #Split each class into training and testing, based on random numbers
        #each class split ensure startification during split
        def per_class(c):
            c = ee.Number(c)
            subset = samples.filter(ee.Filter.eq(class_property, c)) \
                            .randomColumn('random', seed)
            training = subset.filter(
                ee.Filter.Or(
                    ee.Filter.lt('random', threshold),
                    ee.Filter.gte('random', threshold.add(step))
                )
            )
            testing = subset.filter(
                ee.Filter.And(
                    ee.Filter.gte('random', threshold),
                    ee.Filter.lt('random', threshold.add(step))
                )
            )
            return ee.Feature(None, {
                'training': training,
                'validation': testing
            })
        #Applied the splits for each class in the dataset
        splits = classes.map(per_class)
        # merge all classes back together for this fold
        # merged all classes in the training subset
        training = ee.FeatureCollection(splits.map(lambda f: ee.Feature(f).get('training'))).flatten()
        # merge all classes in the testing subset
        testing = ee.FeatureCollection(splits.map(lambda f: ee.Feature(f).get('testing'))).flatten()
        return ee.Feature(None, {'training': training, 'testing': testing})

    folds = thresholds.map(make_fold)
    return ee.FeatureCollection(folds)

def rf_tuning_withkfold(reference_fold, image, class_prop,  
                         n_tree_list, v_split_list, leaf_pop_list, scale = 10, tile_scale = 16):
    """
    Perform parameter optimization for random forest One-vs-rest classification
    with stratified k-fold input data
        parameters: reference_fold: ee.featurecollection result from stratified kfold
        image: ee.image remote sensing data
        n_tree_list: list of int, number of trees to test
        v_split: list of int, number of variables to test
        leaf_pop: list of int, minimum leaf population to test
    return list of dict with parameters and average accuracy
    """    
    #define and set the previous fold result
    k = reference_fold.size().getInfo()
    fold_list = reference_fold.toList(k)
    
    #get the list of unique class id
    first_fold = ee.Feature(fold_list.get(0))
    training_fc0 = ee.FeatureCollection(first_fold.get('training'))
    classes = training_fc0.aggregate_array(class_prop).distinct().getInfo()
    result = []
    #Create a gridsearch tuning by manually looped through the parameter space
    for n_tree in n_tree_list:
        for var_split in v_split_list:
            for min_leaf_pop in leaf_pop_list:
                fold_acc_list = []
                for i in range(k):
                    fold = ee.Feature(fold_list.get(i))

                    training_fc = ee.FeatureCollection(fold.get('training'))
                    testing_fc = ee.FeatureCollection(fold.get('testing'))

                    class_acc = []

                    for c in classes:
                        train_binary = training_fc.map(
                            lambda f: f.set('label', ee.Number(f.get(class_prop)).eq(c))
                        )
                        testing_binary = testing_fc.map(
                            lambda f: f.set('label', ee.Number(f.get(class_prop)).eq(c))
                        )
                        #sample the image 
                        train_pixels = image.sampleRegions(
                            collection = train_binary,
                            properties = ['label'],
                            scale = scale,
                            tileScale = tile_scale
                        )
                        test_pixels = image.sampleRegions(
                            collection = train_binary,
                            properties = ['label'],
                            scale = scale,
                            tileScale = tile_scale
                            )
                        try:
                            #classiifer set to probability for binary one vs rest classification
                            clf = ee.Classifier.smileRandomForest(
                                numberOfTrees = n_tree,
                                variablesPerSplit = var_split,
                                minLeafPopulation=min_leaf_pop,
                                seed=0
                            ).setOutputMode('PROBABILITY').train(
                                features=train_pixels,
                                classProperty = 'label',
                                inputProperties = image.bandNames()
                            )
                            #function to evaluate the model
                            classified_val = test_pixels.classify(clf)
                            model_val = classified_val.errorMatrix('label', 'classification')
                            class_acc.append(model_val.accuracy().getInfo())

                        except Exception as e:
                            print(f"faild for fold {i}, class {c}")
                            print(e)
                    
                    if class_acc: 
                        fold_acc_list.append(sum(class_acc)/len(class_acc))
                    else: 
                        print(f"[WARNING] No valid accuracy scores for fold {i}. Skipping this fold.")
                    
                    if fold_acc_list:
                        avg_acc = sum(fold_acc_list)/len(fold_acc_list)
                    #Put the result into a list
                        result.append({
                            'Number of Trees': n_tree,
                            'variable per split': var_split,
                            'min leaf population': min_leaf_pop,
                            'Average Validation Accuracy': avg_acc
                        })
    
    result_pd = pd.DataFrame(result)
    result_pd_sorted = result_pd.sort_values(by='Average Validation Accuracy', ascending=False).reset_index(drop=True)
    print("Best parameters:\n", result_pd_sorted.iloc[0])            
    return result_pd_sorted
