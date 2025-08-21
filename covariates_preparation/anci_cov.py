import ee
ee.Initialize()
#This is function for calling the ancillary covariates, which consist of terrain and distance metric data

############################# 1. Terrain Metric ###########################
def get_terrain(aoi):
    """
    Return stacked terrain metric, consist of elevation, slope, and aspect data. Further development should used the 
    Indonesian National DEM (DEMNAS), which has higher spatial resolution
    """
    dem = ee.Image("NASA/NASADEM_HGT/001").select('elevation')
    #Units are degrees, range is [0,90).
    slope = ee.Terrain.slope(dem).rename('slope')
    #Units are degrees where 0=N, 90=E, 180=S, 270=W.
    aspect = ee.Terrain.aspect(dem).rename('aspect')
    #stacked the terrain metric
    terrain_metric = ee.Image.cat(dem, slope, aspect)
    return terrain_metric
############################# 2. Distance Metric ###########################
def distance_metric_stack(aoi, max_dist = 500000, in_meters = False):
    """
    This function create a distance image based on predefined dataset (road, coastline, river, etc)
        Parameters:
        aoi (ee.Geometry): Area of interest.
        max_dist (float): Max cumulative cost distance (in meters or pixels).
        in_meters (bool): If True, output distance in meters instead of pixels
    """
    #This script is used when in_meters option are set to True. 
    #However, original script indicate that this process required more computation time
    if in_meters:
        cost_image = ee.Image.pixelArea().sqrt() 
    else:
    #distance in pixel unit, effective storage use
        cost_image = ee.Image(1)
    #This wraps cumulativeCost() code, it did not repeated during the calculation
    def distance_image(source_mask): #Source mask is binary mask, in which value 1 is unmasked data (starting location)
        return cost_image.cumulativeCost(source_mask, max_dist)
    #The Dataset for creating a distance metric, ARE POSSIBLY OUTDATED
    #Natural Earth Coastline data (https://www.naturalearthdata.com/downloads/10m-physical-vectors/10m-coastline/)
    ne_coastline = ee.FeatureCollection(
        "users/hadicu06/IIASA/RESTORE/vector_datasets/coastline_ne_10m"
    ).filterBounds(aoi)
    coast_dist = ne_coastline.map(lambda ft: ft.set('constant', 1)) \
                            .reduceToImage(['constant'], ee.Reducer.first()) \
                            .mask()
    #Roads
    #The original project used both OSM roads layer and RBI, might required update
    road_osm = ee.FeatureCollection(
        "users/hadicu06/IIASA/RESTORE/vector_datasets/road_osm"
    ).filterBounds(aoi)
    road_rbi = ee.FeatureCollection(
        "users/hadicu06/IIASA/RESTORE/vector_datasets/road_rbi"
    ).filterBounds(aoi)
    roads_dist = road_osm.merge(road_rbi) \
                        .map(lambda ft: ft.set('constant', 1)) \
                        .reduceToImage(['constant'], ee.Reducer.max()) \
                        .mask()
    #settlement data from High Resolution Settlement Layer(HRSL) (https://dataforgood.facebook.com/dfg/tools/high-resolution-population-density-maps)
    hrsl = ee.ImageCollection("projects/sat-io/open-datasets/hrsl/hrslpop") \
            .mosaic() \
            .clip(aoi)
    hrsl_connected = hrsl.int().connectedPixelCount(maxSize=100, eightConnected=True)
    hrsl_masked = hrsl_connected.unmask().gt(3)
    #calculate distance metric
    dist_roads =distance_image(roads_dist).rename('dist_roads')
    dist_coast = distance_image(coast_dist).rename('dist_coast')
    dist_settlement = distance_image(hrsl_masked).rename('dist_settlement')
    # Stack into one image
    return ee.Image.cat(dist_roads, dist_coast, dist_settlement)