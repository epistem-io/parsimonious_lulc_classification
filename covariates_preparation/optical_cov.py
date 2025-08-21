import ee
ee.Initialize()

# This function is created to prepared the optical remote sensing data covariates

############################# 1. The Area of Interest Definition ###########################
#Get the Area of Interest, province level
def get_aoi_from_gaul(country="Indonesia", province="Sumatera Selatan"):
    """
    Returns the AOI geometry for a given country and province using the GAUL admin boundaries.
    """
    admin = ee.FeatureCollection("FAO/GAUL/2015/level1")
    aoi_fc = admin.filter(ee.Filter.eq('ADM0_NAME', country)).filter(
        ee.Filter.eq('ADM1_NAME', province)
    )
    return aoi_fc.geometry()
############################# 2. Getting Optical Data Image Collection ###########################
def get_optical_data (aoi, 
                      start_date, 
                      end_date, 
                      optical_data='L8_TOA', 
                      cloud_cover=30, 
                      s2_clear_threshold=0.60):
    """
    Returns optical image collection for Landsat or Sentinel data, with cloud masking.
    Parameters:
    Area of interest(aoi): ee.Geometry or Feature Collection
    start_date: str, e.g. '2023-06-01'
    end_date: str, e.g. '2023-10-30'
    optical_data: Landsat 8 Collection 2 TOA (L8_TOA), Landsat 8 Collection 2 SR (L8_SR)
                 Sentinel-2 L2A SR (S2_SR), Sentinel-2 L1C(S2_TOA)
    cloud_cover: maximum cloud cover coverage, default value is 30%

    """
    if optical_data == 'L8_TOA':
        collection_id = "LANDSAT/LC08/C02/T1_TOA"
    elif optical_data == 'L8_SR':
        collection_id = "LANDSAT/LC08/C02/T1_L2"
    elif optical_data == 'S2_SR':
        collection_id = "COPERNICUS/S2_SR_HARMONIZED"
        csplus_id = "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"
    elif optical_data == 'S2_TOA':
        collection_id = "COPERNICUS/S2_HARMONIZED"
        csplus_id = "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"
    else:
        raise ValueError("Optical Data can only used either: L8_TOA, L8_SR, S2_SR, S2_TOA")
    #Landsat SR Mask
    def mask_landsat_sr(image):
        # Bits for cloud and shadow. Based on QA band
        cloud_shadow_bit_mask = (1 << 4)
        clouds_bit_mask = (1 << 3)
        qa = image.select('QA_PIXEL')
        mask = qa.bitwiseAnd(cloud_shadow_bit_mask).eq(0).And(
            qa.bitwiseAnd(clouds_bit_mask).eq(0)
        )
        # Applied scalling function for landsat collection 2 SR data
        optical_bands = image.select('SR_B.*').multiply(0.0000275).add(-0.2)
        return optical_bands.updateMask(mask).copyProperties(image)
    #Landsat TOA Mask
    def mask_landsat_toa (image):
         #Only used the Quality Bands cloud masking procedure
        cloud_shadow_bit_mask = (1 << 4)
        clouds_bit_mask = (1 << 3)
        qa = image.select('QA_PIXEL')
        mask_qa = qa.bitwiseAnd(cloud_shadow_bit_mask).eq(0).And(
        qa.bitwiseAnd(clouds_bit_mask).eq(0))
        # Apply mask and return image
        return image.updateMask(mask_qa).copyProperties(image)
    # Sentinel-2 Cloud Score+ masking
    def mask_s2_cloudscore(image):
        cs_img = image.select('cs_cdf')
        return image.updateMask(cs_img.gte(s2_clear_threshold))
    # Landsat 8 TOA bands naming
    # renamed the bands to match the requirement of correction
    def band_naming_toa(image) :
        return image.select(['B2', 'B3', 'B4', 'B5', 'B6', 'B7'], ['blue', 'green', 'red', 'nir', 'swir1', 'swir2'])
    #Landsat 8SR bands naming
    def band_naming_sr (image):
        return image.select(['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7'], ['blue', 'green', 'red', 'nir', 'swir1', 'swir2'])
    
 # Build collection depending on dataset
    if optical_data in ['L8_TOA', 'L8_SR']:
        # Landsat 8 TOA
        if optical_data == 'L8_TOA':
            collection = (
                ee.ImageCollection(collection_id)
                .filterBounds(aoi)
                .filterDate(start_date, end_date)
                .filter(ee.Filter.lt('CLOUD_COVER_LAND', cloud_cover))
                .map(mask_landsat_toa)
                .map(band_naming_toa)
            )
        else:  # L8_SR
            collection = (
                ee.ImageCollection(collection_id)
                .filterBounds(aoi)
                .filterDate(start_date, end_date)
                .filter(ee.Filter.lt('CLOUD_COVER_LAND', cloud_cover))
                .map(mask_landsat_sr)
                .map(band_naming_sr)
            )
    else:
        # Sentinel-2 (SR or TOA) with Cloud Score+
        s2 = ee.ImageCollection(collection_id)\
        .filterBounds(aoi).filterDate(start_date, end_date)\
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_cover))
        csplus = ee.ImageCollection(csplus_id)
        collection = (
            s2.linkCollection(csplus, ['cs_cdf'])
            .map(mask_s2_cloudscore)
            .select(['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12'])
        )
    return collection
############################# 3. Temporal Composites ###########################
def get_temporal_composite(collection, aoi, method = 'median', percentile_value=25):
    """
    Create a temporal composites for image collection and clip them based on the AOI
        parameters:
        collection: Image collection from 'get_optical_data' function
        method: ee.Image.reducer() median, mean, std_dev, or percentile (if selected, specified the percentiile value)
        aoi=ee.Image.geometry() or ee.FeatureCollection
    """
    #Define temporal composite Methods
    if method.lower() == 'median':
        img = collection.median()
    elif method.lower() == 'stdDev':
        img = collection.reduce(ee.Reducer.stdDev())
    elif method.lower() == 'Percentile':
        img = collection.reduce(ee.Reducer.percentile([percentile_value]))
    elif method.lower() == 'mean':
        img = collection.mean()
    else:
        raise ValueError("Method must be 'median', 'mean', 'stdDev', or 'percentile'")
    return img.clip(aoi)

############################# 4. BRDF Correction ###########################

# Bidirectional Reflectance Distribution Function (BRDF) script adapted from Poortinga et al 2019 (https://doi.org/10.3390/rs11070831)
# further reading Berra et al. 2024 (https://doi.org/10.3390/rs16152695)
# value for parameters, copied from the original script
PI = ee.Number(3.14159265359)
MAX_SATELLITE_ZENITH = 7.5
MAX_DISTANCE = 1000000
UPPER_LEFT = 0
LOWER_LEFT = 1
LOWER_RIGHT = 2
UPPER_RIGHT = 3

#function to applied BRDF correction 
def apply_brdf(image):
    date = ee.Date(image.get('system:time_start')) #image acqusition for solar position calculation
    footprint = ee.List(image.geometry().bounds().bounds().coordinates().get(0)) #required for sensor geometry calculation
    sunAz, sunZen = getsunAngles(date, footprint) 
    viewAz = azimuth(footprint) 
    viewZen = zenith(footprint)
    kvol, kvol0 = _kvol(sunAz, sunZen, viewAz, viewZen) #BRDF Kernels
    result = _apply(image, kvol.multiply(PI), kvol0.multiply(PI))
    return result

#Function for calculating solar azimuth (direction) and zenith (angle from vertical) per pixel
def getsunAngles(date, footprint):
    jdp = date.getFraction('year') #Julian Day fraction + pixel latlon
    seconds_in_hour = 3600
    hourGMT = ee.Number(date.getRelative('second', 'day')).divide(seconds_in_hour)
    latRad = ee.Image.pixelLonLat().select('latitude').multiply(PI.divide(180))
    longDeg = ee.Image.pixelLonLat().select('longitude')

    jdpr = jdp.multiply(PI).multiply(2)
    a = ee.List([0.000075, 0.001868, 0.032077, 0.014615, 0.040849])

    meanSolarTime = longDeg.divide(15.0).add(hourGMT)
    localSolarDiff1 = (value(a, 0)
        .add(value(a, 1).multiply(jdpr.cos()))
        .subtract(value(a, 2).multiply(jdpr.sin()))
        .subtract(value(a, 3).multiply(jdpr.multiply(2).cos()))
        .subtract(value(a, 4).multiply(jdpr.multiply(2).sin())))

    localSolarDiff2 = localSolarDiff1.multiply(12 * 60)
    localSolarDiff = localSolarDiff2.divide(PI)
    trueSolarTime = meanSolarTime.add(localSolarDiff.divide(60)).subtract(12.0)

    ah = trueSolarTime.multiply(ee.Number(MAX_SATELLITE_ZENITH * 2).multiply(PI.divide(180)))
    b = ee.List([0.006918, 0.399912, 0.070257, 0.006758, 0.000907, 0.002697, 0.001480])
    delta = (value(b, 0)
        .subtract(value(b, 1).multiply(jdpr.cos()))
        .add(value(b, 2).multiply(jdpr.sin()))
        .subtract(value(b, 3).multiply(jdpr.multiply(2).cos()))
        .add(value(b, 4).multiply(jdpr.multiply(2).sin()))
        .subtract(value(b, 5).multiply(jdpr.multiply(3).cos()))
        .add(value(b, 6).multiply(jdpr.multiply(3).sin())))

    cosSunZen = latRad.sin().multiply(delta.sin()).add(latRad.cos().multiply(ah.cos()).multiply(delta.cos()))
    sunZen = cosSunZen.acos()

    sinSunAzSW = ah.sin().multiply(delta.cos()).divide(sunZen.sin()).clamp(-1.0, 1.0)
    cosSunAzSW = ((latRad.cos().multiply(-1).multiply(delta.sin()))
                  .add(latRad.sin().multiply(delta.cos()).multiply(ah.cos()))
                  ).divide(sunZen.sin())
    sunAzSW = sinSunAzSW.asin()
    sunAzSW = where(cosSunAzSW.lte(0), sunAzSW.multiply(-1).add(PI), sunAzSW)
    sunAzSW = where(cosSunAzSW.gt(0).And(sinSunAzSW.lte(0)), sunAzSW.add(PI.multiply(2)), sunAzSW)

    sunAz = sunAzSW.add(PI)
    sunAz = where(sunAz.gt(PI.multiply(2)), sunAz.subtract(PI.multiply(2)), sunAz)

    footprint_polygon = ee.Geometry.Polygon(footprint)
    sunAz = sunAz.clip(footprint_polygon).rename(['sunAz'])
    sunZen = sunZen.clip(footprint_polygon).rename(['sunZen'])

    return ee.Image(sunAz), ee.Image(sunZen)
#estimating sensor viewing azimuth/viewing direction
def azimuth(footprint):
    def x(point): return ee.Number(ee.List(point).get(0))
    def y(point): return ee.Number(ee.List(point).get(1))
    upperCenter = line_from_coords(footprint, UPPER_LEFT, UPPER_RIGHT).centroid().coordinates()
    lowerCenter = line_from_coords(footprint, LOWER_LEFT, LOWER_RIGHT).centroid().coordinates()
    slope = (y(lowerCenter).subtract(y(upperCenter))).divide(x(lowerCenter).subtract(x(upperCenter)))
    slopePerp = ee.Number(-1).divide(slope)
    azimuthLeft = ee.Image(PI.divide(2).subtract(slopePerp.atan()))
    return azimuthLeft.rename(['viewAz'])

#estimaying sensor vertical view/off angle nadir
def zenith(footprint):
    leftLine = line_from_coords(footprint, UPPER_LEFT, LOWER_LEFT)
    rightLine = line_from_coords(footprint, UPPER_RIGHT, LOWER_RIGHT)
    leftDistance = ee.FeatureCollection(leftLine).distance(MAX_DISTANCE)
    rightDistance = ee.FeatureCollection(rightLine).distance(MAX_DISTANCE)
    viewZenith = (rightDistance.multiply(ee.Number(MAX_SATELLITE_ZENITH * 2))
                  .divide(rightDistance.add(leftDistance))
                  .subtract(ee.Number(MAX_SATELLITE_ZENITH))
                  .clip(ee.Geometry.Polygon(footprint))
                  .rename(['viewZen']))
    return viewZenith.multiply(PI.divide(180))

#implement the BRDF for each bands using predefined values using volumetric kernel (kvol/ Ross–Thick model)
    # http://www.mdpi.com/2072-4292/9/12/1325/htm#sec3dot2-remotesensing-09-01325 (literature for Sentinel)
    # https://www.sciencedirect.com/science/article/pii/S0034425717302791
def _apply(image, kvol, kvol0):
    blue_cor = _correct_band(image, 'blue', kvol, kvol0, 0.0774, 0.0079, 0.0372)
    green_cor = _correct_band(image, 'green', kvol, kvol0, 0.1306, 0.0178, 0.0580)
    red_cor = _correct_band(image, 'red', kvol, kvol0, 0.1690, 0.0227, 0.0574)
    nir_cor = _correct_band(image, 'nir', kvol, kvol0, 0.3093, 0.0330, 0.1535)
    swir1_cor = _correct_band(image, 'swir1', kvol, kvol0, 0.3430, 0.0453, 0.1154)
    swir2_cor = _correct_band(image, 'swir2', kvol, kvol0, 0.2658, 0.0387, 0.0639)
    return image.select([]).addBands([blue_cor, green_cor, red_cor, nir_cor, swir1_cor, swir2_cor])

#Helper function
def _correct_band(image, band_name, kvol, kvol0, f_iso, f_geo, f_vol):
    iso = ee.Image(f_iso)
    geo = ee.Image(f_geo)
    vol = ee.Image(f_vol)
    pred = vol.multiply(kvol).add(geo.multiply(kvol)).add(iso)
    pred0 = vol.multiply(kvol0).add(geo.multiply(kvol0)).add(iso)
    cfac = pred0.divide(pred)
    corr = image.select(band_name).multiply(cfac).rename([band_name])
    return corr


def _kvol(sunAz, sunZen, viewAz, viewZen):
    relative_azimuth = sunAz.subtract(viewAz)
    pa1 = viewZen.cos().multiply(sunZen.cos())
    pa2 = viewZen.sin().multiply(sunZen.sin()).multiply(relative_azimuth.cos())
    phase_angle1 = pa1.add(pa2)
    phase_angle = phase_angle1.acos()
    p1 = ee.Image(PI.divide(2)).subtract(phase_angle)
    p2 = p1.multiply(phase_angle1)
    p3 = p2.add(phase_angle.sin())
    p4 = sunZen.cos().add(viewZen.cos())
    p5 = ee.Image(PI.divide(4))
    kvol = p3.divide(p4).subtract(p5)

    viewZen0 = ee.Image(0)
    pa10 = viewZen0.cos().multiply(sunZen.cos())
    pa20 = viewZen0.sin().multiply(sunZen.sin()).multiply(relative_azimuth.cos())
    phase_angle10 = pa10.add(pa20)
    phase_angle0 = phase_angle10.acos()
    p10 = ee.Image(PI.divide(2)).subtract(phase_angle0)
    p20 = p10.multiply(phase_angle10)
    p30 = p20.add(phase_angle0.sin())
    p40 = sunZen.cos().add(viewZen0.cos())
    p50 = ee.Image(PI.divide(4))
    kvol0 = p30.divide(p40).subtract(p50)
    return kvol, kvol0

#utility function
def line_from_coords(coords, fromIndex, toIndex):
    return ee.Geometry.LineString(ee.List([coords.get(fromIndex), coords.get(toIndex)]))


def where(condition, trueValue, falseValue):
    trueMasked = trueValue.mask(condition)
    falseMasked = falseValue.mask(invertMask(condition))
    return trueMasked.unmask(falseMasked)


def invertMask(mask):
    return mask.multiply(-1).add(1)


def value(lst, idx):
    return ee.Number(lst.get(idx))



############################# 5. Topographic/Illumination Correction ###########################
#The script is adapted from poortinga et al 2019 (https://doi.org/10.3390/rs11070831)
def terrain_correction(collection):
    """
    Applies terrain correction to a collection of images using the Sun-Canopy-Sensor + C (SCSc) method.
    This function first calculates the illumination condition (IC) for each image and then applies the
    SCSc correction to the specified bands. The IC is a measure of the amount of direct solar radiation
    received by a surface, which is affected by both the solar position and the terrain's slope and aspect.
    The SCSc method then normalizes the reflectance values based on this illumination condition to
    reduce the effects of topography on the imagery.

    Params:
        collection (ee.ImageCollection): The input image collection to be corrected.

    Output:
        ee.ImageCollection: The terrain-corrected image collection.
    """
    # Terrain layer source. Note: NASADEM is used since Chukwuwa et al 2024 (https://doi.org/10.1080/10095020.2023.2296010)
    # found that it have improve vertical accuracy compared to other dem product
    dem = ee.Image("NASA/NASADEM_HGT/001")
    #Function to calculate illumination condition (IC). Function by Patrick Burns(pb463@nau.edu) and Matt Macander (mmacander@abrinc.com)
    def illumination_condition(img):
        """
        Calculates the illumination condition (IC) for an image.
        """
        # Extract image metadata about solar position
        # in recent update of Landsat data in GEE, Solar Zenit Angle are renamed into 'SUN_ELEVATION'
        SZ_rad = ee.Image.constant(ee.Number(img.get('SUN_ELEVATION'))).multiply(3.14159265359).divide(180).clip(img.geometry().buffer(10000))
        # Solar azimuth angle are renamed into 'SUN_AZIMUTH'
        SA_rad = ee.Image.constant(ee.Number(img.get('SUN_AZIMUTH')).multiply(3.14159265359).divide(180)).clip(img.geometry().buffer(10000))
        # Create slope and aspect layer
        slp = ee.Terrain.slope(dem).clip(img.geometry().buffer(10000))
        slp_rad = ee.Terrain.slope(dem).multiply(3.14159265359).divide(180).clip(img.geometry().buffer(10000))
        asp_rad = ee.Terrain.aspect(dem).multiply(3.14159265359).divide(180).clip(img.geometry().buffer(10000))

        # Calculate the Illumination Condition (IC)
        # Slope part of the illumination condition
        cosZ = SZ_rad.cos()
        cosS = slp_rad.cos()
        slope_illumination = cosZ.multiply(cosS)

        # Aspect part of the illumination condition
        sinZ = SZ_rad.sin()
        sinS = slp_rad.sin()
        cosAziDiff = (SA_rad.subtract(asp_rad)).cos()
        aspect_illumination = sinZ.multiply(sinS).multiply(cosAziDiff)

        # Full illumination condition (IC)
        ic = slope_illumination.add(aspect_illumination)

        # Add IC to original image
        img_plus_ic = img.addBands(ic.rename('IC')).addBands(cosZ.rename('cosZ')).addBands(cosS.rename('cosS')).addBands(slp.rename('slope'))
        return img_plus_ic

    def illumination_correction(img):
        """
        Applies the Sun-Canopy-Sensor + C (SCSc) correction method to each image.
        """
        props = img.toDictionary()
        st = img.get('system:time_start')
        
        img_plus_ic = img
        mask2 = img_plus_ic.select('slope').gte(5).And(img_plus_ic.select('IC').gte(0)).And(img_plus_ic.select('nir').gt(-0.1))
        img_plus_ic_mask2 = img_plus_ic.updateMask(mask2)

        # Specify Bands to topographically correct  
        band_list = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']
        composite_bands = img.bandNames()
        
        def apply_SCSccorr(band):
            band = ee.String(band)
            reducer = ee.Reducer.linearRegression(2, 1)
            out = ee.Image(1).addBands(img_plus_ic_mask2.select(['IC', band])) \
                .reduceRegion(
                    reducer=reducer,
                    geometry=img.geometry(),
                    scale=300,
                    bestEffort=True,
                    maxPixels=1e10
                )
            # Handle cases where regression fails (fill defaults)
            fit = out.combine({"coefficients": ee.Array([[1],[1]])}, False)
            out_a = ee.Array(fit.get('coefficients')).get([0, 0])
            out_b = ee.Array(fit.get('coefficients')).get([1, 0])
            out_c = out_a.divide(out_b)

            # Apply the SCSc correction
            SCSc_output = img_plus_ic_mask2.expression(
                "((image * (cosB * cosZ + cvalue)) / (ic + cvalue))",
                {
                    'image': img_plus_ic_mask2.select(band),
                    'ic': img_plus_ic_mask2.select('IC'),
                    'cosB': img_plus_ic_mask2.select('cosS'),
                    'cosZ': img_plus_ic_mask2.select('cosZ'),
                    'cvalue': out_c
                }
            )
            return SCSc_output.rename(band)
        
        corrected_bands =  [apply_SCSccorr(band) for band in band_list]
        img_SCSccorr = ee.Image.cat(corrected_bands) \
                        .addBands(img_plus_ic.select('IC')) \
                        .unmask(img_plus_ic.select(band_list + ['IC'])) \
                        .select(band_list)
        return img_SCSccorr

    collection = collection.map(illumination_condition)
    collection = collection.map(illumination_correction)

    return collection
############################# 6. Spectral Transformation ###########################
def cal_indices(image):
    """
    Return stacks of spectral transformation indices as listed below:
    1. Normalized Difference Vegetation Index (NDVI)
    2. Normalized Difference Moisture Index (NDMI)
    3. Modified Soil Adjusted Vegetation Index (MSAVI)
    4. Normalized Burn Ratio (NBR)
    5. Modified Normalized Difference Water Index (MNDWI)
    6. Renormalization of Vegetation Moisture Index (RVMI)
    7. Tasseled Cap Transformation (Brightness, Grenness, and Wetness)
    """
    #calculate the spectral transformation
    ndvi = image.normalizedDifference(['nir', 'red']).rename('NDVI')
    ndmi = image.normalizedDifference(['nir', 'swir1']).rename('NDMI')
    nbr = image.normalizedDifference(['nir', 'swir2']).rename('NBR')
    msavi = image.expression(
    '(2 * NIR + 1 - sqrt((2 * NIR + 1) ** 2 - 8 * (NIR - RED))) / 2',
    {
        'NIR': image.select('nir'),
        'RED': image.select('red')
    }
    ).rename('MSAVI')

    mndwi = image.normalizedDifference(['green', 'swir1']).rename('MNDWI')
    rvmi = ndvi.subtract(ndmi).divide(ndvi.add(ndmi)).rename('RVMI')
    evi = image.expression(
    '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))',
    {
        'BLUE': image.select('blue'),
        'NIR': image.select('nir'),
        'RED': image.select('red')
    }
    ).rename('EVI')


    b = image.select(["blue", "green", "red", "nir", "swir1", "swir2"])
    #Coefficients are only for Landsat 8  derived from Zhai et al (2022) (https://doi.org/10.1016/j.rse.2022.112992)
    brightness_coefficents= ee.Image([0.3443, 0.4057, 0.4667, 0.5347, 0.3936, 0.2412])
    greenness_coefficents= ee.Image([-0.2365, -0.2836, -0.4257,	0.8097,	0.0043,	-0.1638])
    wetness_coefficents= ee.Image([0.1301, 0.2280,	0.3492,	0.1795,	-0.6270, -0.6195])
        #Calculate tasseled cap transformation
    brightness = b.multiply(brightness_coefficents).reduce(ee.Reducer.sum()).rename('brightness')
    greenness  = b.multiply(greenness_coefficents).reduce(ee.Reducer.sum()).rename('greenness')
    wetness    = b.multiply(wetness_coefficents).reduce(ee.Reducer.sum()).rename('wetness')
    tasseled_cap = brightness.addBands([greenness, wetness])
        #stacked them into a single imagery
    spectral_indices = ee.Image.cat(ndvi, ndmi, nbr, msavi, mndwi, rvmi, evi, tasseled_cap)
    return spectral_indices
    