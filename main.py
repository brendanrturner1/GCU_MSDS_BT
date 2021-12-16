#!/usr/bin/env python
# coding: utf-8

# In[45]:


import cv2
import numpy as np
from PIL import Image, ImageOps
import numpy as np 
import pandas as pd 
import keras 
import tensorflow as tf
from skimage import io, color
import imutils
import PySimpleGUIWeb as sg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# In[7]:


from tensorflow import keras
model = keras.models.load_model(r'C:\Users\giant\hasyv2-dataset-friend-of-mnist\HASYv2\maths_model.h5')


# In[2]:


def letter_width(contours):
	letter_width_sum = 0
	count = 0
	for cnt in contours:
		if cv2.contourArea(cnt) > 20:
			x,y,w,h = cv2.boundingRect(cnt)
			letter_width_sum += w
			count += 1

	return letter_width_sum/count


# In[5]:


def sort_contours(cnts, method="left-to-right"):
    reverse = False
    i = 0
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
    key=lambda b:b[1][i], reverse=reverse))
    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)


# In[12]:


def segment_image_to_lines(file):
    '''Collects a file and segments the file into lines'''
    image = cv2.imread(file)
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)#grayscale scale image
    blur = cv2.GaussianBlur(gray,(35,35),cv2.BORDER_DEFAULT)#Application of Blur
    th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,21,10)#Binarization
    kernel = np.ones((5,200), np.uint8)
    img_dilation = cv2.dilate(th3, kernel, iterations=1)#dilation
    #find and save the contours
    cnts = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sort_contours(cnts, method="top-to-bottom")[0]
    # loop over the contours
    lines = []
    for c in cnts:
        # compute the bounding box of the contour
        (x, y, w, h) = cv2.boundingRect(c)
        # filter out bounding boxes, ensuring they are neither too small
        # nor too large
        if (w >= 300) and (h >= 80):
            # extract the line then grab the width and height of the line's image
            roi = gray[y:y + h, x:x + w]
            lines.append((roi, (x, y, w, h)))
    return lines


# In[39]:


def segment_line_to_chars(line_input):
    '''Segment a line into various characters'''
    l_blurred = cv2.GaussianBlur(line_input, (5, 5),cv2.BORDER_DEFAULT)#blur the line
    l_edged = cv2.Canny(l_blurred,30,150, 0)#find the edges
    (_, l_thresh) = cv2.threshold(l_edged, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)#Binarization
    kernel = np.ones((5,5), np.uint8) #Dilation
    l_dilated = cv2.dilate(l_thresh,kernel,iterations = 2)
    #find contours
    l_cnts = cv2.findContours(l_dilated.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    l_cnts = imutils.grab_contours(l_cnts)
    l_cnts = sort_contours(l_cnts, method="left-to-right")[0]
    chars=[]
    # loop over the contours
    for c1 in l_cnts:
        # compute the bounding box of the contour
        (x1, y1, w1, h1) = cv2.boundingRect(c1)
        # filter out bounding boxes, ensuring they are not too small
        if (w1 >= 20) and (h1 >= 40):
            # extract the character and threshold it to make the character
            # appear as *white* (foreground) on a *black* background, then
            # grab the width and height of the thresholded image
            l_roi = line_input[y1:y1 + h1, x1:x1 + w1]
            l_thresh = cv2.threshold(l_roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            (tH, tW) = l_thresh.shape
                # if the width is greater than the height, resize along the
            # width dimension
            if tW > tH:
                l_thresh = imutils.resize(l_thresh, width=28)
                # otherwise, resize along the height
            else:
                l_thresh = imutils.resize(l_thresh, height=28)
                        # re-grab the image dimensions (now that its been resized)
            # and then determine how much we need to pad the width and
            # height such that our image will be 28x28
            (tH, tW) = l_thresh.shape
            dX = int(max(0, 32 - tW) / 2.0)
            dY = int(max(0, 32 - tH) / 2.0)

            # pad the image and force 28x28 dimensions
            padded = cv2.copyMakeBorder(l_thresh, top=dY, bottom=dY, left=dX, right=dX, 
                                        borderType=cv2.BORDER_CONSTANT,value=(0, 0, 0))
            padded = cv2.resize(padded, (32, 32))

            # prepare the padded image for classification via our
            # handwriting OCR model
            padded = padded.astype("float32") / 255.0
            padded = np.expand_dims(padded, axis=-1)
            padded = np.repeat(padded,3,-1)
            # update our list of characters that will be OCR'd
            chars.append((padded, (x1, y1, w1, h1)))
    return chars


# In[24]:


c = ['0','1','2','3','4','5','6','7','8','9','\\pi','\\frown','\\smile',
    '+','-','/','|',']','[','=','\\ast','\\bullet',
    '\\leq', '\\geq', '<', '>','\\div','\\times',
    'A', 'B', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M',
    'N', 'Q', 'R', 'T', 'U', 'V', 'Y', 'Z','\\cdot',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'j', 'k', 'm',
    'n', 'p', 'q', 'r', 's','u', 'v', 'w', 'x', 'y', 'z']
j=[]
for i in range(1,len(c)):
    j.append(i-1)
label_dict = dict(zip(j,c))


# In[40]:


def ocr_predict(chars):
    '''Predict the Characters'''
    chars = [c[0] for c in chars]
    boxes = [b[1] for b in chars]
    chars = np.array(chars)
    print(chars.shape)
    pred = model.predict(chars)
    pred = np.apply_along_axis(np.argmax, 1, pred)
    pred = [label_dict[val+1] for val in pred]
    predicted_chars = ' '.join(pred)
    return predicted_chars


# In[41]:


def test_marker(script):
    lines = segment_image_to_lines(script)
    predicted_spellings = []
    for i in range(len(lines)):
        chars = segment_line_to_chars(lines[i][0])
        ocr_predict(chars)
        predicted_chars = ocr_predict(chars)
        predicted_spellings.append(predicted_chars)
    print(predicted_spellings)


# In[ ]:


from sympy.abc import *
from sympy import solve
from sympy.parsing.sympy_parser import parse_expr

def solve_meThis(string_):
    try:
        lhs =  parse_expr(string_.split("=")[0])
        rhs =  parse_expr(string_.split("=")[1])
        solution = solve(lhs-rhs)
        return solution
    except:
        print("invalid equation")

def solver(operation):
    def operate(fb, sb, op):
        result = 0
        if operator == '+':
            result = int(first_buffer) + int(second_buffer)
        elif operator == '-':
            result = int(first_buffer) - int(second_buffer)
        elif operator == 'x':
            result = int(first_buffer) * int(second_buffer)
        return result

    if not operation or not operation[0].isdigit():
        return -1

    operator = ''
    first_buffer = ''
    second_buffer = ''

    for i in range(len(operation)):
        if operation[i].isdigit():
            if len(second_buffer) == 0 and len(operator) == 0:
                first_buffer += operation[i]
            else:
                second_buffer += operation[i]
        else:
            if len(second_buffer) != 0:
                result = operate(first_buffer, second_buffer, operator)
                first_buffer = str(result)
                second_buffer = ''
            operator = operation[i]

    result = int(first_buffer)
    if len(second_buffer) != 0 and len(operator) != 0:
        result = operate(first_buffer, second_buffer, operator)

    return result

def calculate(operation):
    string,head = '', None
    temp = string = str(operation)
    if 'D' in string:
        string = string.replace('D', '0')
    if 'G' in string:
        string = string.replace('G', '6')
    if 'b' in string:
        string = string.replace('b', '6')
    if 'B' in string:
        string = string.replace('B', '8')
    if 'Z' in string:
        string = string.replace('Z', '2')
    if 'S' in string:
        string = string.replace('S', '=')
    if 't' in string:
        string = string.replace('t', '+')
    if 'f' in string:
        string = string.replace('f', '7')
    if 'M' in string:
        string = string.replace('M', '-')
    if 'W' in string:
        string = string.replace('W', '-')
    if 'L' in string:
        string = string.replace('L', '/')
    if 'g' in string:
        string = string.replace('g', '9')
    if '=' not in string:
        if 'x' in string:
            string = string.replace('x', '*')
        if 'X' in string:
            string = string.replace('X', '*')
        return string, eval(string)
        
    operation = string
    string = ''
    for k in operation:
        if head is None:
            head = k
            string += head
        if k in ['+', '-', '*', '/', '%', '^', '='] or head in ['+', '-', '*', '/', '%', '^', '=']:
            head = k
            string += head
        elif k.isnumeric() and not head.isnumeric():
            head = k
            added = '**' + k
            string += added
        elif not k.isnumeric() and head.isnumeric():
            head = k
            added = '*' + k
            string += added
        
    
    print(string)
    if '=' not in string:
        return string, solver(string)
    else:
        return string, solve_meThis(string)


# In[47]:


# Define Main GUI
def main():
    sg.theme('LightBlue')

    layout = [[sg.Text('Welcome!')],
              [sg.Text("Import Image"), sg.Input(), sg.FileBrowse(key="-IN-")],
              [[sg.Canvas(size=(100,100), key='-CANVAS1-', pad=(15,15))]],
              [sg.B("Solution: "), key = 'sol']]

    win = sg.Window('BT Capstone', layout)

    while True:  # Event Loop
        event, values = win.read()
        if event == sg.WIN_CLOSED:
            break
        elif event == "Import Image":
            im = values["-IN-"]
            canvas_elem = get_im(im)
            canvas_elem = win['-CANVAS-'].TKCanvas
            canvas_elem.Size=(int(get_im[1]),int(get_im[2]))
            seg = segment_image_to_lines(im)
            res = test_marker(seg)
            ans = calculate(res)
            win.Element('sol').update(values=ans)
            
    win.close()


# In[ ]:


main()

