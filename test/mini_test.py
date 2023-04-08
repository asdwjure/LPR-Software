import easyocr


reader = easyocr.Reader(['en'], gpu=False)
results = reader.readtext('test.jpg')
print(results)
