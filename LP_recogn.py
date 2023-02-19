import cv2
import imutils
import numpy as np
import pytesseract

pytesseract.pytesseract.tesseract_cmd = 'C:\Program Files\Tesseract-OCR\\tesseract.exe' # Tesseract path on windows (NOTE: For Linux this must be deleted)


def clean2_plate(plate):
    gray_img = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray_img, 110, 255, cv2.THRESH_BINARY)
    num_contours,hierarchy = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if num_contours:
        contour_area = [cv2.contourArea(c) for c in num_contours]
        max_cntr_index = np.argmax(contour_area)

        max_cnt = num_contours[max_cntr_index]
        max_cntArea = contour_area[max_cntr_index]
        x,y,w,h = cv2.boundingRect(max_cnt)

        if not ratioCheck(max_cntArea,w,h):
            return plate,None

        final_img = thresh[y:y+h, x:x+w]
        return final_img,[x,y,w,h]

    else:
        return plate,None

def ratioCheck(area, width, height):
    ratio = float(width) / float(height)
    if ratio < 1:
        ratio = 1 / ratio
    if (area < 1063.62 or area > 73862.5) or (ratio < 3 or ratio > 6):
        return False
    return True

def isMaxWhite(plate):
    avg = np.mean(plate)
    if(avg>=115):
        return True
    else:
        return False

def ratio_and_rotation(rect):
    (x, y), (width, height), rect_angle = rect

    if(width>height):
        angle = -rect_angle
    else:
        angle = 90 + rect_angle

    if angle>15:
        return False

    if height == 0 or width == 0:
        return False

    area = height*width
    if not ratioCheck(area,width,height):
        return False
    else:
        return True


def straighten_plate(img, rect):
    rect = cv2.minAreaRect(rect)
    (x,y), (w,h), angle = rect

    if h>w:
        angle -= 90

    rect = (x,y), (w,h), angle
        

    box = cv2.boxPoints(rect)
    #srcPts = [[x,y], [x+w,y], [x+w,y+h], [x,y+h]]
    srcPts = box
    dstPts = [[0, h], [0,0], [w, 0], [w, h]]

    H = cv2.getPerspectiveTransform(np.float32(srcPts), np.float32(dstPts))
    out = cv2.warpPerspective(img, H, (int(w), int(h)), flags=cv2.INTER_LINEAR)
    return out





if __name__ == "__main__":

    img = cv2.imread('images/slo3.jpg', cv2.IMREAD_COLOR)
    img = cv2.resize(img, (600,400) )

    cv2.imshow('Original slika', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    #gray = cv2.bilateralFilter(gray, 13, 15, 15)

    cv2.imshow('Zglajena siva', gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    edged = cv2.Canny(gray, 30, 200)

    cv2.imshow('Robovi', edged)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]
    screenCnt = None

    img_cnt = img.copy()
    cv2.drawContours(img_cnt, contours, -1, (0,255,0), 1)

    cv2.imshow('Konture', img_cnt)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    for c in contours:
        
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)    # najdi koliko tock najbolje opise konturo, da je pri obsegu manj kot 0,18% odstopanja
    
        if len(approx) == 4:
            img_cnt = img.copy()
            cv2.drawContours(img_cnt, approx, -1, (0,0,255), 3)
            cv2.imshow('Posamezne konture', img_cnt)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            x,y,w,h = cv2.boundingRect(c)
            plate_img = img.copy()[y:y+h,x:x+w]

            plate_img = straighten_plate(img, approx)

            cv2.imshow('Tablica', plate_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            screenCnt = approx
            break

    if screenCnt is None:
        detected = 0
        print ("No contour detected")
    else:
        detected = 1

    if detected == 1:
        cv2.drawContours(img, [screenCnt], -1, (0, 0, 255), 3)

    mask = np.zeros(gray.shape,np.uint8)
    new_image = cv2.drawContours(mask,[screenCnt],0,255,-1,)
    new_image = cv2.bitwise_and(img,img,mask=mask)

    (x, y) = np.where(mask == 255)
    (topx, topy) = (np.min(x), np.min(y))
    (bottomx, bottomy) = (np.max(x), np.max(y))
    Cropped = gray[topx:bottomx+1, topy:bottomy+1]

    text = pytesseract.image_to_string(Cropped, config='--psm 11')
    print("programming_fever's License Plate Recognition\n")
    print("Detected license plate Number is:",text)
    img = cv2.resize(img,(500,300))
    Cropped = cv2.resize(Cropped,(400,200))
    cv2.imshow('car',img)
    cv2.imshow('Cropped',Cropped)

    cv2.waitKey(0)
    cv2.destroyAllWindows()