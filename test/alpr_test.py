import sys
from openalpr import Alpr
# from vehicleclassifier import VehicleClassifier

COUNTRY = 'us'
IMAGE_PATH = 'slo1.jpg'

# Initialize instances
alpr = Alpr(COUNTRY, '/path/to/openalpr.conf', '/path/to/runtime_data')
if not alpr.is_loaded():
    print('Error loading Alpr')
    sys.exit(1)
# vehicle = VehicleClassifier('/path/to/openalpr.conf', '/path/to/runtime_data')
# if not vehicle.is_loaded():
#     print('Error loading VehicleClassifier')
#     sys.exit(1)

# Set additional detection parameters (optional)
alpr.set_top_n(5)
alpr.set_default_region('md')

# Gather and print results
plate_results = alpr.recognize_file(IMAGE_PATH)
for i, plate in enumerate(plate_results['results']):
    print('Plate {:-<30}'.format(i))
    for c in plate['candidates']:
        display = '\t{:>7} {}'.format('{:.2f}%'.format(c['confidence']), c['plate'])
        if c['matches_template']:
            display += ' *'
        print(display)

# vehicle_results = vehicle.recognize_file(COUNTRY, IMAGE_PATH)
# best = [v[0]['name'] for k, v in  vehicle_results.items() if k != 'make']
# print('\nTop vehicle: {} oriented at {} degrees'.format(' '.join(best[:-1]), best[-1]))
# for attribute, candidates in vehicle_results.items():
#     print('\n{:-<30}'.format(attribute.capitalize()))
#     for c in candidates:
#         label = c['name']
#         if c['name'] == 'missing':
#             label = 'unknown'
#         print('\t{:>7} {}'.format('{:.2f}%'.format(c['confidence']), label))

# Call when completely done to release memory
alpr.unload()
# vehicle.unload()