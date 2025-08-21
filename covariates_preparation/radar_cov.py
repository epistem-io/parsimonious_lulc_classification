import ee
ee.Initialize()
import sys
sys.path.append(r"C:\Users\62853\Documents\GitHub\Rubber_case_study\gee_s1_ard\python-api")
import s1_ard
from wrapper import s1_preproc
# This function is created to prepared the radar remote sensing data covariates
############################# 1. ALOS PALSAR ###########################
def get_palsar_data(aoi, start_date, end_date,
                    polarizations=['HH', 'HV']):
    """
    return the PALSAR backscatter data GEE and returns image collection. Applied the speckle filter using 
    hydrafloods library
    Parameters:
        aoi (ee.Geometry): Area of interest.
        start_date (str): Start date in 'YYYY-MM-DD'.
        end_date (str): End date in 'YYYY-MM-DD'.
        version: chose between ScanSAR data or yearly mosaic.
        polarizations (list): List of polarization strings to select (e.g. ['HH', 'HV']).
        reducer: choose image reducer 
    note: The ScanSAR polarization band (either HH or HV) is not avaliable in certain location and time frame. 
    If This happen, choose the Yearly Mosaic
    """
    palsar_coll = (
                ee.ImageCollection('JAXA/ALOS/PALSAR-2/Level2_2/ScanSAR')
                .filterBounds(aoi)
                .filterDate(start_date, end_date)
                .select(polarizations))

    def dn_to_gamma0(img):
        gamma0 = (
            img.pow(2)
               .log10()
               .multiply(10)
               .subtract(83.0)
        )
        return gamma0.copyProperties(img, img.propertyNames())

    gamma0_coll = palsar_coll.map(dn_to_gamma0)

    return gamma0_coll
############################# 2. Sentinel-1 ###########################
def get_s1_data(aoi, 
                start_date, end_date, 
                orbit = 'DESCENDING',
                polarisations = 'VV',
                speckle_filter = 'Refined Lee',
                resolution = 10,
                clip_to_aoi = True):
    """
    return the Sentinel-1 GRD Backscatter data and preprocessed them according to 
    Mullissa, A.; Vollrath, A.; Odongo-Braun, C.; Slagter, B.; Balling, J.; Gou,  Y.; Gorelick, N.; Reiche, J. 
    Sentinel-1 SAR Backscatter Analysis Ready Data Preparation in Google Earth Engine. 
    Remote Sens. 2021, 13, 1954. https://doi.org/10.3390/rs13101954
    Parameters:
        aoi (ee.Geometry): Area of interest.
        start_date (str): Start date in 'YYYY-MM-DD'.
        end_date (str): End date in 'YYYY-MM-DD'.
        polarisations (list): List of polarisation bands to include ['VV', 'VH'] or ['VV'].
        orbit (str): 'ASCENDING' or 'DESCENDING' (optional).
        speckle_filter (str): Speckle filter type (''BOXCAR','LEE','GAMMA MAP','REFINED LEE','LEE SIGMA').
        resolution (int): Output resolution in meters.
        clip_to_aoi (bool): If True, clip output images to AOI.
    """
    # Set ARD parameters
    parameters = {
        'APPLY_BORDER_NOISE_CORRECTION': True,
        'APPLY_TERRAIN_FLATTENING': True,
        'APPLY_SPECKLE_FILTERING': True,
        'POLARIZATION': polarisations,   # 'VV', 'VH', or 'VVVH'
        'ORBIT': orbit,                  # 'ASCENDING', 'DESCENDING', or 'BOTH'
        'SPECKLE_FILTER_FRAMEWORK': 'MULTI',  # monotemporal or multitemporal
        'SPECKLE_FILTER': speckle_filter.upper(),  # Speckle filter method. must match repo names: 'BOXCAR','LEE','GAMMA MAP','REFINED LEE','LEE SIGMA'
        'SPECKLE_FILTER_KERNEL_SIZE': 7,
        'SPECKLE_FILTER_NR_OF_IMAGES': 10,
        'TERRAIN_FLATTENING_MODEL': 'VOLUME',
        'DEM': ee.Image('USGS/SRTMGL1_003'),
        'TERRAIN_FLATTENING_ADDITIONAL_LAYOVER_SHADOW_BUFFER': 0,
        'FORMAT': 'DB',   # 'LINEAR' or 'DB'
        'START_DATE': start_date,
        'STOP_DATE': end_date,
        'ROI': aoi,
        'CLIP_TO_ROI': clip_to_aoi,
        'SAVE_ASSET': False,   # don’t export
        'ASSET_ID': None
    }

        # Run the preprocessing'
    s1_ready = s1_preproc(parameters)
    return s1_ready
############################# 3. Annual Composite ###########################
def radar_annual_comp(aoi, collection, reducer='median', percentile_value = 50):
    """
    return the already preprocessed radar data and create an annual composite based on desired reducer
        parameters:
        collection: The imagery collection already preprocessed (Sentinel-1 or ALOS PALSAR)
        reducer: ee.Reducer
        percentile_value = only applied if percentile reducer is used
    """
    reducer = reducer.lower()
    if reducer == 'median':
        comp = collection.median()
    elif reducer == 'mean':
        comp = collection.mean()
    elif reducer == 'std_Dev':
        comp = collection.reduce(ee.Reducer.stdDev())
    elif reducer == 'percentile':
        comp = collection.reduce(ee.Reducer.percentile([percentile_value]))
    else:
        raise ValueError (f"unsupported reducer type:{reducer}")
    composite_clip = comp.clip(aoi)
    return composite_clip
 ############################# 4. Seasonal Composite ###########################\
 #for the seasonal composite we used Climate Hazards Center InfraRed Precipitation With Station Data (CHIRPS)
 #rainfall data to automatically identify the dry and wet season. We used that information to determine the dry and wet season
 #alternatively, the user can mannually input the month range
def s1_seasonal_composite (s1_coll, aoi, start_date, end_date,
                           precip_threshold = 170, #Example value for Sumatera Selatan, specified for better classification
                           rainy_month = None, #specified rainy months if CHIRPS data is not used
                           dry_month = None, #specified dry months if CHIRPS data is not used
                           reducer = 'median',
                           percentile_value = 50):
    """
    return sentinel-1 seasonal composite, in which the user can choose to input the dry and wet season or used
    CHIRPS data to automatically determine the dry and wet season
    """
    #If the rainy and dry month is set to 'None'
    if rainy_month is None or dry_month is None:
            #Get the CHIRPS 
            chirps = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
            months = ee.List.sequence(1, 12)
            #function to calculate total rainfall in each month
            def monthly_precip(m):
                m = ee.Number(m)
                monthly_sum = chirps.filterBounds(aoi) \
                                    .filterDate(start_date, end_date) \
                                    .filter(ee.Filter.calendarRange(m, m, 'month')) \
                                    .sum() \
                                    .reduceRegion( #Reduce to a single number for the AOI
                                        reducer=ee.Reducer.mean(), 
                                        geometry=aoi,
                                        scale=5566,  # ~5.6 km
                                        maxPixels=1e13
                                    ).get('precipitation')
                return ee.Feature(None, {'month': m, 'rain_mm':monthly_sum}) #create a features with two data, month and total rainfall
            #Build feature collection of average total rainfall in the AOI
            rain_stats = ee.FeatureCollection(months.map(monthly_precip)) 
            #Print the list of average rainfall in each month
            rain_stats_list = rain_stats.getInfo()
            for feature in rain_stats_list['features']:
                month = feature['properties']['month']
                avg_rain = feature['properties']['rain_mm']
                print(f"Month {month}: Avg Rainfall = {avg_rain:.2f} mm")
            #Split the rainy and dry season based on precipitation threshold
            rainy_month = rain_stats.filter(ee.Filter.gt('rain_mm', precip_threshold)) \
                                 .aggregate_array('month').getInfo()
            dry_month = rain_stats.filter(ee.Filter.lte('rain_mm', precip_threshold)) \
                               .aggregate_array('month').getInfo()
            if not rainy_month or not dry_month:
                raise ValueError("Unable to determine rainy/dry months — adjust threshold or automatically detect them using CHIRPS data")
    def seasonal_reduction(month_list, suffix):
        season = s1_coll.filter(ee.Filter.inList('month', month_list))
        if reducer == 'median':
            comp = season.median()
        elif reducer == 'mean':
            comp = season.mean()
        elif reducer == 'std_dev':
            comp =  season.reduce(ee.Reducer.stdDev())
        elif reducer == 'percentile': 
            comp = season.reduce(ee.Reducer.percentile([percentile_value]))
        else:
            return ValueError('Unable to use the reducer, avaliable option are: median, mean, std_dev, and percentile')
        new_names = comp.bandNames().map(lambda b: ee.String(b).cat(f"_{suffix}"))
        return comp.rename(new_names)
    rainy_comp = seasonal_reduction(rainy_month, "rainy")
    dry_comp = seasonal_reduction(dry_month, "dry")
    return ee.Image.cat(rainy_comp, dry_comp).clip(aoi)
