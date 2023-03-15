from licenseplaterecognition import LicensePlateRecognition
import multiprocessing
import sys
sys.path.append('/home/jrebernik/Magistrska/LPR-Software/webapp')
import webapp

PATH_TO_MODEL='/home/jrebernik/Magistrska/LPR-Software/RPi/detect.tflite'
PLATE_CHARS = 'ABCDEFGHIJKLMNOPRSTUVZYXQ1234567890' # All possible chars in a plate
PLATE_CITIES = ['KP', 'LJ', 'KR', 'GO', 'PO', 'NM', 'MB', 'SG', 'KK', 'MS', 'CE']

def image_processing(q):
    lpr.process_image(q)

def webapp_process(q):
    webapp.start(q)

if __name__ == '__main__':

    # Create an instance (object) of LicensePlateRecognition class
    params = {
        'model_path' : PATH_TO_MODEL,
        'capture_source' : '/home/jrebernik/Magistrska/LPR-Software/dataset/collected_images/test_video2.avi',
        'min_conf_threshold' : 0.8,
        'min_area_plate' : 0.05,
        'min_laplacian_var' : 25,
        'plate_chars': 'ABCDEFGHIJKLMNOPRSTUVZYXQ1234567890',
        'plate_cities' : PLATE_CITIES
    }

    q = multiprocessing.Queue(maxsize=5)

    lpr = LicensePlateRecognition(params=params, debug_level=0)

    # Start license plate image processing in a separate process
    p1 = multiprocessing.Process(target = image_processing, args=(q,))
    p2 = multiprocessing.Process(target = webapp_process, args=(q,))
    p1.start()
    p2.start()
    p1.join()

    
