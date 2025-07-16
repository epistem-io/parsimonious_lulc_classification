import ee

def get_aoi_from_gaul(country="Indonesia", province="Sumatera Selatan"):
    """
    Returns the AOI geometry for a given country and province using the GAUL admin boundaries.
    """
    admin = ee.FeatureCollection("FAO/GAUL/2015/level1")
    aoi_fc = admin.filter(ee.Filter.eq('ADM0_NAME', country)).filter(
        ee.Filter.eq('ADM1_NAME', province)
    )
    return aoi_fc.geometry()

def get_landsat_composite(
    aoi,
    start_date,
    end_date,
    landsat_version='LC09',
    cloud_cover=20,
    bands=['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7']
):
    """
    Returns a median composite image for Landsat 8 or 9 Collection 2 SR, with cloud masking.
    Parameters:
        aoi: ee.Geometry or ee.FeatureCollection
        start_date: str, e.g. '2023-06-01'
        end_date: str, e.g. '2023-10-30'
        landsat_version: 'LC08' for Landsat 8, 'LC09' for Landsat 9
        cloud_cover: int, max cloud cover percentage
        bands: list of bands to select for composite
        limit: int, max number of images to use
    Returns:
        ee.Image (median composite, clipped to AOI)
    """
    if landsat_version == 'LC08':
        collection_id = 'LANDSAT/LC08/C02/T1_L2'
    elif landsat_version == 'LC09':
        collection_id = 'LANDSAT/LC09/C02/T1_L2'
    else:
        raise ValueError("landsat_version must be 'LC08' or 'LC09'")

    def mask_landsat_sr(image):
        # Bits for cloud and shadow
        cloud_shadow_bit_mask = (1 << 4)
        clouds_bit_mask = (1 << 3)
        qa = image.select('QA_PIXEL')
        mask = qa.bitwiseAnd(cloud_shadow_bit_mask).eq(0).And(
            qa.bitwiseAnd(clouds_bit_mask).eq(0)
        )
        optical_bands = image.select('SR_B.*').multiply(0.0000275).add(-0.2)
        return optical_bands.updateMask(mask).copyProperties(image, ["system:time_start"])

    collection = (
        ee.ImageCollection(collection_id)
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt('CLOUD_COVER', cloud_cover))
        .map(mask_landsat_sr)
    )

    composite = collection.select(bands).median().clip(aoi)
    return composite

def add_spectral_indices(image):
    """
    Adds NDVI, NDWI, NBR, SAVI, and EVI2 bands to a Landsat 8/9 image.
    Assumes input image has SR_B2, SR_B3, SR_B4, SR_B5, SR_B6, SR_B7.
    Returns image with new bands.
    """
    # NDVI
    ndvi = image.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')
    # NDWI
    ndwi = image.normalizedDifference(['SR_B5', 'SR_B6']).rename('NDWI')
    # NBR
    nbr = image.normalizedDifference(['SR_B5', 'SR_B7']).rename('NBR')
    # SAVI
    savi = image.expression(
        '(1 + L) * ((NIR - RED) / (NIR + RED + L))', {
            'NIR': image.select('SR_B5'),
            'RED': image.select('SR_B4'),
            'L': 0.9
        }).rename('SAVI')
    # EVI2
    evi2 = image.expression(
        '2.5 * ((NIR - RED) / (NIR + 2.4 * RED + 1))', {
            'NIR': image.select('SR_B5'),
            'RED': image.select('SR_B4')
        }).rename('EVI2')
    return image.addBands([ndvi, ndwi, nbr, savi, evi2])

def split_training_validation(points_fc, split=0.7, seed=42):
    """
    Splits a FeatureCollection into training and validation sets using a random column.
    Args:
        points_fc: ee.FeatureCollection, input points with class property.
        split: float, fraction for training set (e.g., 0.7 means 70% training, 30% validation).
        seed: int, random seed for reproducibility.
    Returns:
        (training_fc, validation_fc): tuple of ee.FeatureCollection
    """
    with_random = points_fc.randomColumn('random', seed)
    training = with_random.filter(ee.Filter.lt('random', split))
    validation = with_random.filter(ee.Filter.gte('random', split))
    return training, validation

def sample_composite(composite, points_fc, bands, class_property='kelas', scale=60, tileScale=4):
    """
    Samples the composite image at the locations of the points FeatureCollection.
    Args:
        composite: ee.Image, input composite with indices.
        points_fc: ee.FeatureCollection, points to sample.
        bands: list of str, band names to sample.
        class_property: str, property name for class label.
        scale: int, sampling scale in meters.
        tileScale: int, tileScale parameter for memory optimization.
    Returns:
        ee.FeatureCollection with sampled values.
    """
    return composite.select(bands).sampleRegions(
        collection=points_fc,
        properties=[class_property],
        scale=scale,
        tileScale=tileScale,
        geometries=True
    )

# ===== NEW FUNCTIONS FOR FLEXIBLE TRAINING POINTS HANDLING =====

def get_aoi_from_gaul_regency(country="Indonesia", province="Sumatera Selatan", regency="Banyuasin"):
    """
    Returns the AOI geometry for a given country, province, and regency using the GAUL admin boundaries.
    
    Args:
        country: str, country name (default: "Indonesia")
        province: str, province/state name (default: "Sumatera Selatan")
        regency: str, regency/district name (default: "Banyuasin")
        
    Returns:
        ee.Geometry, geometry of the specified regency
        
    Raises:
        Exception: If no matching regency is found
    """
    try:
        # Use GAUL level 2 for regency/district boundaries
        admin = ee.FeatureCollection("FAO/GAUL/2015/level2")
        
        # Filter by country, province, and regency
        aoi_fc = admin.filter(ee.Filter.eq('ADM0_NAME', country)) \
                     .filter(ee.Filter.eq('ADM1_NAME', province)) \
                     .filter(ee.Filter.eq('ADM2_NAME', regency))
        
        # Check if any features were found
        size = aoi_fc.size()
        if size.getInfo() == 0:
            raise Exception(f"No regency found with name '{regency}' in {province}, {country}. "
                          f"Please check the spelling and try again.")
        
        print(f"[SUCCESS] Found AOI: {regency}, {province}, {country}")
        return aoi_fc.geometry()
        
    except Exception as e:
        raise Exception(f"Error getting regency boundaries: {str(e)}")

def validate_training_points(points_fc, aoi_geometry):
    """
    Validates a training points FeatureCollection for basic requirements.
    
    Args:
        points_fc: ee.FeatureCollection, training points to validate
        aoi_geometry: ee.Geometry, area of interest geometry
        
    Returns:
        dict: Validation results with 'valid' (bool) and 'message' (str)
        
    Raises:
        Exception: If validation fails with critical errors
    """
    try:
        # Check if FeatureCollection exists and has features
        size = points_fc.size()
        total_points = size.getInfo()
        
        if total_points == 0:
            raise Exception("Training points FeatureCollection is empty.")
        
        # Check geometry type (should be points)
        first_feature = ee.Feature(points_fc.first())
        geom_type = first_feature.geometry().type()
        
        if geom_type.getInfo() != 'Point':
            raise Exception(f"Training points must be Point geometries, found: {geom_type.getInfo()}")
        
        # Check CRS (should be EPSG:4326 or similar geographic coordinate system)
        crs = points_fc.first().geometry().projection().crs()
        crs_info = crs.getInfo()
        
        if not crs_info.startswith('EPSG:4326'):
            print(f"[WARNING] Training points CRS is {crs_info}, expected EPSG:4326")
        
        # Check spatial coverage within AOI
        points_in_aoi = points_fc.filterBounds(aoi_geometry)
        points_in_aoi_count = points_in_aoi.size().getInfo()
        
        if points_in_aoi_count == 0:
            raise Exception("No training points found within the specified Area of Interest. "
                          "Please check that your training points overlap with the AOI.")
        
        coverage_percentage = (points_in_aoi_count / total_points) * 100
        
        print(f"[SUCCESS] Training points validation passed:")
        print(f"  - Total points: {total_points}")
        print(f"  - Points in AOI: {points_in_aoi_count} ({coverage_percentage:.1f}%)")
        print(f"  - Geometry type: Point")
        print(f"  - CRS: {crs_info}")
        
        return {
            'valid': True,
            'message': f"Validation successful. {points_in_aoi_count} points available in AOI.",
            'points_in_aoi': points_in_aoi_count,
            'total_points': total_points,
            'coverage_percentage': coverage_percentage
        }
        
    except Exception as e:
        raise Exception(f"Training points validation failed: {str(e)}")

def analyze_class_distribution(points_fc, class_property='kelas', min_points_per_class=10):
    """
    Analyzes the class distribution in training points and provides warnings if needed.
    
    Args:
        points_fc: ee.FeatureCollection, training points
        class_property: str, property name containing class labels (default: 'kelas')
        min_points_per_class: int, minimum points per class for warning (default: 10)
        
    Returns:
        dict: Class distribution analysis results
    """
    try:
        # Get unique classes and their counts
        class_counts = points_fc.aggregate_histogram(class_property)
        class_counts_dict = class_counts.getInfo()
        
        total_points = sum(class_counts_dict.values())
        num_classes = len(class_counts_dict)
        
        # Check for classes with insufficient points
        insufficient_classes = []
        for class_id, count in class_counts_dict.items():
            if count < min_points_per_class:
                insufficient_classes.append(f"Class {class_id}: {count} points")
        
        print(f"[INFO] Class Distribution Analysis:")
        print(f"  - Total classes: {num_classes}")
        print(f"  - Total points: {total_points}")
        print(f"  - Average points per class: {total_points/num_classes:.1f}")
        
        # Display class counts
        for class_id, count in sorted(class_counts_dict.items()):
            percentage = (count / total_points) * 100
            print(f"  - Class {class_id}: {count} points ({percentage:.1f}%)")
        
        # Warnings for insufficient classes
        if insufficient_classes:
            print(f"\n[WARNING] {len(insufficient_classes)} classes have fewer than {min_points_per_class} points:")
            for class_info in insufficient_classes:
                print(f"    - {class_info}")
            print("  This may affect classification performance for these classes.")
        else:
            print(f"\n[SUCCESS] All classes have at least {min_points_per_class} points.")
        
        return {
            'class_counts': class_counts_dict,
            'total_points': total_points,
            'num_classes': num_classes,
            'insufficient_classes': insufficient_classes,
            'min_points_per_class': min_points_per_class
        }
        
    except Exception as e:
        print(f"[WARNING] Could not analyze class distribution: {str(e)}")
        return {'error': str(e)}

def get_training_points_for_aoi(aoi_geometry, user_training_points_asset=None, 
                               backup_training_points_asset='projects/ee-rg2icraf/assets/Sumsel_GT_Restore',
                               class_property='kelas', min_points_per_class=10):
    """
    Main function to get training points for a given AOI with automatic fallback logic.
    
    Args:
        aoi_geometry: ee.Geometry, area of interest geometry
        user_training_points_asset: str or None, user-provided training points asset path
        backup_training_points_asset: str, backup training points asset path
        class_property: str, property name containing class labels (default: 'kelas')
        min_points_per_class: int, minimum points per class for warnings (default: 10)
        
    Returns:
        ee.FeatureCollection: Validated training points clipped to AOI
        
    Raises:
        Exception: If no valid training points can be obtained
    """
    try:
        training_points = None
        source_info = ""
        
        # Step 1: Try user-provided asset first
        if user_training_points_asset is not None:
            print(f"[INFO] Attempting to use user-provided training points: {user_training_points_asset}")
            try:
                user_points = ee.FeatureCollection(user_training_points_asset)
                validate_training_points(user_points, aoi_geometry)
                
                # Clip to AOI
                training_points = user_points.filterBounds(aoi_geometry)
                source_info = f"user-provided asset: {user_training_points_asset}"
                print(f"[SUCCESS] Successfully loaded user training points")
                
            except Exception as e:
                print(f"[ERROR] Failed to use user training points: {str(e)}")
                print(f"[INFO] Falling back to backup training points...")
        
        # Step 2: Use backup asset if user asset failed or wasn't provided
        if training_points is None:
            print(f"[INFO] Using backup training points: {backup_training_points_asset}")
            try:
                backup_points = ee.FeatureCollection(backup_training_points_asset)
                validate_training_points(backup_points, aoi_geometry)
                
                # Clip to AOI
                training_points = backup_points.filterBounds(aoi_geometry)
                source_info = f"backup asset: {backup_training_points_asset}"
                print(f"[SUCCESS] Successfully loaded backup training points")
                
            except Exception as e:
                raise Exception(f"Failed to load backup training points: {str(e)}")
        
        # Step 3: Final validation of clipped points
        final_count = training_points.size().getInfo()
        if final_count == 0:
            raise Exception("No training points remain after clipping to AOI. "
                          "Please check that your AOI overlaps with available training data.")
        
        # Step 4: Analyze class distribution
        print(f"\n[INFO] Final training points summary:")
        print(f"  - Source: {source_info}")
        print(f"  - Points in AOI: {final_count}")
        
        # Analyze class distribution and show warnings
        analyze_class_distribution(training_points, class_property, min_points_per_class)
        
        return training_points
        
    except Exception as e:
        raise Exception(f"Failed to get training points for AOI: {str(e)}")