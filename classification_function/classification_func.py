import ee
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
                         n_tree_list, v_split_list, leaf_pop_list):
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
    k = reference_fold.size().get.Info()
    fold_list = reference_fold.toList(k)
    #get the list of unique class id
    classes = ee.FeatureCollection(fold_list.get(0)).aggregate_array('training').distinct().getInfo()
    result = []
    for n_tree in n_tree_list:
        for var_split in v_split_list:
            for min_leaf_pop in leaf_pop_list:
                fold_acc_list = []
                for i in range(k):
                    fold = ee.Feature(fold_list.get(i))

                    training_fc = ee.FeatureCOllection(fold.get('training'))
                    testing_fc = ee.FeatureCollection(fold.get('testing'))

                    class_acc = []

                    for c in classes:
                        train_binary = training_fc.map(
                            lambda f: f.set('label', ee.Number(f.get(class_prop)).eq(c))
                        )
                        testing_binary = testing_fc.map(
                            lambda f: f.set('label', ee.Number(f.get(class_prop)).eq(c))
                        )
                        try:
                            #classiifer
                            clf = ee.Classifier.smileRandomForest(
                                numberOfTrees = n_tree,
                                variablesPerSplit = var_split,
                                minLeafPopulation=min_leaf_pop,
                                seed=0
                            ).setOutputMode('PROBABILITY').train(
                                features=train_binary,
                                classProperty = 'label',
                                inputProperties = image.bandNames()
                            )
                        except Exception as e:
                            print(f"faild for fold {i}, class {c}")
                            print(e)
                    fold_acc_list.append(sum(class_acc)/len(class_acc))
                avg_acc = sum(fold_acc_list)/len(fold_acc_list)

                result.append({
                    'Number of Trees': n_tree,
                    'variable per split': var_split,
                    'min leaf population': min_leaf_pop,
                    'Average Validation Accuracy': avg_acc
                })
    return result