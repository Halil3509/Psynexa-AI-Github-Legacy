import cv2 
import numpy as np

FONTS =cv2.FONT_HERSHEY_COMPLEX

# colors 
# values =(blue, green, red) opencv accepts BGR values not RGB
BLACK = (0,0,0)
WHITE = (255,255,255)
BLUE = (255,0,0)
RED = (0,0,255)
CYAN = (255,255,0)
YELLOW =(0,255,255)
MAGENTA = (255,0,255)
GRAY = (128,128,128)
GREEN = (0,255,0)
PURPLE = (128,0,128)
ORANGE = (0,165,255)
PINK = (147,20,255)
points_list =[(200, 300), (150, 150), (400, 200)]
def drawColor(img, colors):
    x, y = 0,10
    w, h = 20, 30
    
    for color in colors:
        x += w+5 
        # y += 10 
        cv2.rectangle(img, (x-6, y-5 ), (x+w+5, y+h+5), (10, 50, 10), -1)
        cv2.rectangle(img, (x, y ), (x+w, y+h), color, -1)
    

def textWithBackground(img, text, font, fontScale, textPos, textThickness=1,textColor=(0,255,0), bgColor=(0,0,0), pad_x=3, pad_y=3, bgOpacity=0.5):
    """
    Draws text with background, with  control transparency
    @param img:(mat) which you want to draw text
    @param text: (string) text you want draw
    @param font: fonts face, like FONT_HERSHEY_COMPLEX, FONT_HERSHEY_PLAIN etc.
    @param fontScale: (double) the size of text, how big it should be.
    @param textPos: tuple(x,y) position where you want to draw text
    @param textThickness:(int) fonts weight, how bold it should be
    @param textPos: tuple(x,y) position where you want to draw text
    @param textThickness:(int) fonts weight, how bold it should be.
    @param textColor: tuple(BGR), values -->0 to 255 each
    @param bgColor: tuple(BGR), values -->0 to 255 each
    @param pad_x: int(pixels)  padding of in x direction
    @param pad_y: int(pixels) 1 to 1.0 (), controls transparency of  text background 
    @return: img(mat) with draw with background
    """
    (t_w, t_h), _= cv2.getTextSize(text, font, fontScale, textThickness) # getting the text size
    x, y = textPos
    overlay = img.copy() # coping the image
    cv2.rectangle(overlay, (x-pad_x, y+ pad_y), (x+t_w+pad_x, y-t_h-pad_y), bgColor,-1) # draw rectangle 
    new_img = cv2.addWeighted(overlay, bgOpacity, img, 1 - bgOpacity, 0) # overlaying the rectangle on the image.
    cv2.putText(new_img,text, textPos,font, fontScale, textColor,textThickness ) # draw in text
    img = new_img

    return img


def textBlurBackground(img, text, font, fontScale, textPos, textThickness=1,textColor=(0,255,0),kneral=(33,33) , pad_x=3, pad_y=3):
    """    
    Draw text with background blured,  control the blur value, with kernal(odd, odd)
    @param img:(mat) which you want to draw text
    @param text: (string) text you want draw
    @param font: fonts face, like FONT_HERSHEY_COMPLEX, FONT_HERSHEY_PLAIN etc.
    @param fontScale: (double) the size of text, how big it should be.
    @param textPos: tuple(x,y) position where you want to draw text
    @param textThickness:(int) fonts weight, how bold it should be.
    @param textColor: tuple(BGR), values -->0 to 255 each
    @param kneral: tuple(3,3) int as odd number:  higher the value, more blurry background would be
    @param pad_x: int(pixels)  padding of in x direction
    @param pad_y: int(pixels)  padding of in y direction
    @return: img mat, with text drawn, with background blured
    
    call the function: 
     img =textBlurBackground(img, 'Blured Background Text', cv2.FONT_HERSHEY_COMPLEX, 0.9, (20, 60),2, (0,255, 0), (49,49), 13, 13 )
    """
    
    (t_w, t_h), _= cv2.getTextSize(text, font, fontScale, textThickness) # getting the text size
    x, y = textPos
    blur_roi = img[y-pad_y-t_h: y+pad_y, x-pad_x:x+t_w+pad_x] # croping Text Background
    img[y-pad_y-t_h: y+pad_y, x-pad_x:x+t_w+pad_x]=cv2.blur(blur_roi, kneral)  # merging the blured background to img
    cv2.putText(img,text, textPos,font, fontScale, textColor,textThickness )          
    # cv2.imshow('blur roi', blur_roi)
    # cv2.imshow('blured', img)

    return img

def fillPolyTrans(img, points, color, opacity):
    """
    @param img: (mat) input image, where shape is drawn.
    @param points: list [tuples(int, int) these are the points custom shape,FillPoly
    @param color: (tuples (int, int, int)
    @param opacity:  it is transparency of image.
    @return: img(mat) image with rectangle draw.

    """
    list_to_np_array = np.array(points, dtype=np.int32)
    overlay = img.copy()  # coping the image
    cv2.fillPoly(overlay,[list_to_np_array], color )
    new_img = cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0)
    # print(points_list)
    img = new_img
    cv2.polylines(img, [list_to_np_array], True, color,1, cv2.LINE_AA)
    return img

def rectTrans(img, pt1, pt2, color, thickness, opacity):
    """

    @param img: (mat) input image, where shape is drawn.
    @param pt1: tuple(int,int) it specifies the starting point(x,y) os rectangle
    @param pt2: tuple(int,int)  it nothing but width and height of rectangle
    @param color: (tuples (int, int, int), it tuples of BGR values
    @param thickness: it thickness of board line rectangle, if (-1) passed then rectangle will be fulled with color.
    @param opacity:  it is transparency of image.
    @return:
    """
    overlay = img.copy()
    cv2.rectangle(overlay, pt1, pt2, color, thickness)
    new_img = cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0) # overlaying the rectangle on the image.
    img = new_img

    return img

def landmarksDetection(img, results):
    img_height, img_width= img.shape[:2]
    # list[(x,y), (x,y)....]
    # mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in results.multi_face_landmarks[0].landmark]
    mesh_points=np.array([np.multiply([p.x, p.y], [img_width, img_height]).astype(int) for p in results.multi_face_landmarks[0].landmark])

    # returning the list of tuples for each landmarks 
    return mesh_points


