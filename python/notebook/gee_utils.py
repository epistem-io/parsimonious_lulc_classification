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