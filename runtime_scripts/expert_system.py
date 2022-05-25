# Libraries
from pyueye import ueye
import numpy as np
import cv2
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from pyModbusTCP.client import ModbusClient
from time import sleep
import math
import os
import time
import random
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# ---------------------------------------------------------------------------------------------------------------------------------------
# Function def
def detection_pattern(img):
    global centers_now
    image_ori = cv2.GaussianBlur(img, (3, 3), 0)

    # Try template Matching
    res = cv2.matchTemplate(image_ori, template, cv2.TM_CCOEFF_NORMED)  # research types of matching SQDIFF mocnena diference
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)  # min max values and location in image
    #print(max_loc)
    centers_now = (max_loc[0] + int(w / 2), max_loc[1] + int(h / 2))
    #print(max_val)
    if max_val > 0.4:
        template_found = True
    else:
        template_found = False

    return [max_loc, template_found, centers_now]

def create_hue_mask(image, lower, upper):
    mask = cv2.inRange(image, lower, upper)
    M = cv2.moments(mask)
    try:
        # calculate x,y coordinate of center
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    except ZeroDivisionError:
        cX = 0
        cY = 0
    return mask, np.sum(mask == 255), [cX, cY]

# ---------------------------------------------------------------------------------------------------------------------------------------
# Variables
# KALMAN
A = np.array([[1.0000, 0.0220, 0, 0],[0, 1.0000, 0, 0],[0, 0, 1.0000, 0.0220],[0, 0, 0, 1.0000]])
C = np.array([[1,0.011000000000000,0,0],[0,0,1,0.011000000000000]])
Q = np.diag([1e-3, 1e1, 1e-3, 1e1]);
R = np.zeros([1,2])
R = np.array([1e-5,1e-5])

x_est = np.array([[0], [0], [0], [0]])
P_est = np.diag([10, 10, 10, 10]);

#conection init
client = ModbusClient("147.228.125.34", port = 502)
client.open()
#modbus init
client.write_multiple_registers(0, [400, 0, 525, 0, 1, 1, 0, 0, 0])

# threshold
white_threshold = np.array([0, 0, 90, 188, 91, 255], np.uint8)
pink_threshold = np.array([152, 133, 72, 255, 255, 255], np.uint8)
yellow_threshold = np.array([0, 204, 30, 40, 255, 255], np.uint8)

#rot_init var
frame_counter = 0
size_eps = 10

golie_size = [] # array for calibration
def_size = []

# ball template
template = cv2.imread('pattern.png', 0)  # template for matching
kernel = np.ones((3, 3), np.uint8)  # size of kernel for dilatation and erosion
w, h = template.shape[::-1]

# INIT Camera
hCam = ueye.HIDS(0)  # 0: first available camera;  1-254: The camera with the specified camera ID
sInfo = ueye.SENSORINFO()
cInfo = ueye.CAMINFO()
pcImageMemory = ueye.c_mem_p()
MemID = ueye.int()
rectAOI = ueye.IS_RECT()
pitch = ueye.INT()
nBitsPerPixel = ueye.INT(24)  # 24: bits per pixel for color mode; take 8 bits per pixel for monochrome
channels = 3  # 3: channels for color mode(RGB); take 1 channel for monochrome
m_nColorMode = ueye.INT()  # Y8/RGB16/RGB24/REG32
bytes_per_pixel = int(nBitsPerPixel / 8)
# ---------------------------------------------------------------------------------------------------------------------------------------
print("START")
print()

# Starts the driver and establishes the connection to the camera
nRet = ueye.is_InitCamera(hCam, None)
if nRet != ueye.IS_SUCCESS:
    print("is_InitCamera ERROR")

# test set params from ini file
pParam = ueye.wchar_p()
pParam.value = "ids_camera5_low_res.ini"
nRet = ueye.is_ParameterSet(hCam, ueye.IS_PARAMETERSET_CMD_LOAD_FILE, pParam, 0)
if nRet != ueye.IS_SUCCESS:
    print("is_ParamSet ERROR")

# Reads out the data hard-coded in the non-volatile camera memory and writes it to the data structure that cInfo points to
nRet = ueye.is_GetCameraInfo(hCam, cInfo)
if nRet != ueye.IS_SUCCESS:
    print("is_GetCameraInfo ERROR")

# You can query additional information about the sensor type used in the camera
nRet = ueye.is_GetSensorInfo(hCam, sInfo)
if nRet != ueye.IS_SUCCESS:
    print("is_GetSensorInfo ERROR")

# Set display mode to DIB
nRet = ueye.is_SetDisplayMode(hCam, ueye.IS_SET_DM_DIB)

# Set the right color mode
if int.from_bytes(sInfo.nColorMode.value, byteorder='big') == ueye.IS_COLORMODE_BAYER:
    # setup the color depth to the current windows setting
    ueye.is_GetColorDepth(hCam, nBitsPerPixel, m_nColorMode)
    bytes_per_pixel = int(nBitsPerPixel / 8)
    print("IS_COLORMODE_BAYER: ", )
    print("\tm_nColorMode: \t\t", m_nColorMode)
    print("\tnBitsPerPixel: \t\t", nBitsPerPixel)
    print("\tbytes_per_pixel: \t\t", bytes_per_pixel)
    print()

elif int.from_bytes(sInfo.nColorMode.value, byteorder='big') == ueye.IS_COLORMODE_CBYCRY:
    # for color camera models use RGB32 mode
    m_nColorMode = ueye.IS_CM_BGRA8_PACKED
    nBitsPerPixel = ueye.INT(32)
    bytes_per_pixel = int(nBitsPerPixel / 8)
    print("IS_COLORMODE_CBYCRY: ", )
    print("\tm_nColorMode: \t\t", m_nColorMode)
    print("\tnBitsPerPixel: \t\t", nBitsPerPixel)
    print("\tbytes_per_pixel: \t\t", bytes_per_pixel)
    print()

elif int.from_bytes(sInfo.nColorMode.value, byteorder='big') == ueye.IS_COLORMODE_MONOCHROME:
    # for color camera models use RGB32 mode
    m_nColorMode = ueye.IS_CM_MONO8
    nBitsPerPixel = ueye.INT(8)
    bytes_per_pixel = int(nBitsPerPixel / 8)
    print("IS_COLORMODE_MONOCHROME: ", )
    print("\tm_nColorMode: \t\t", m_nColorMode)
    print("\tnBitsPerPixel: \t\t", nBitsPerPixel)
    print("\tbytes_per_pixel: \t\t", bytes_per_pixel)
    print()

else:
    # for monochrome camera models use Y8 mode
    m_nColorMode = ueye.IS_CM_MONO8
    nBitsPerPixel = ueye.INT(8)
    bytes_per_pixel = int(nBitsPerPixel / 8)
    print("else")

# Can be used to set the size and position of an "area of interest"(AOI) within an image
nRet = ueye.is_AOI(hCam, ueye.IS_AOI_IMAGE_GET_AOI, rectAOI, ueye.sizeof(rectAOI))
if nRet != ueye.IS_SUCCESS:
    print("is_AOI ERROR")

width = rectAOI.s32Width
height = rectAOI.s32Height

# Prints out some information about the camera and the sensor
print("Camera model:\t\t", sInfo.strSensorName.decode('utf-8'))
print("Camera serial no.:\t", cInfo.SerNo.decode('utf-8'))
print("Maximum image width:\t", width)
print("Maximum image height:\t", height)
print()

# ---------------------------------------------------------------------------------------------------------------------------------------

# Allocates an image memory for an image having its dimensions defined by width and height and its color depth defined by nBitsPerPixel
nRet = ueye.is_AllocImageMem(hCam, width, height, nBitsPerPixel, pcImageMemory, MemID)
if nRet != ueye.IS_SUCCESS:
    print("is_AllocImageMem ERROR")
else:
    # Makes the specified image memory the active memory
    nRet = ueye.is_SetImageMem(hCam, pcImageMemory, MemID)
    if nRet != ueye.IS_SUCCESS:
        print("is_SetImageMem ERROR")
    else:
        # Set the desired color mode
        nRet = ueye.is_SetColorMode(hCam, m_nColorMode)

# Activates the camera's live video mode (free run mode)
nRet = ueye.is_CaptureVideo(hCam, ueye.IS_DONT_WAIT)
if nRet != ueye.IS_SUCCESS:
    print("is_CaptureVideo ERROR")

# Enables the queue mode for existing image memory sequences
nRet = ueye.is_InquireImageMem(hCam, pcImageMemory, MemID, width, height, nBitsPerPixel, pitch)
if nRet != ueye.IS_SUCCESS:
    print("is_InquireImageMem ERROR")
else:
    print("Press q to leave the programm")

# ---------------------------------------------------------------------------------------------------------------------------------------
# Start calibration

print("Starting motors")
print(client.write_single_register(6, 1))

defender_bool = client.read_holding_registers(11, 1)
while defender_bool[0] == 0:
    defender_bool = client.read_holding_registers(11, 1)
    sleep(0.2)

print("Defender Wait")
sleep(2)
print("Defender Start")

while (nRet == ueye.IS_SUCCESS):
    frame_counter += 1
    array = ueye.get_data(pcImageMemory, width, height, nBitsPerPixel, pitch, copy=False)
    # ...reshape it in an numpy array...
    frame = np.reshape(array, (height.value, width.value, bytes_per_pixel))
    # defender
    if defender_bool:
        frame_def = frame[200:height.value - 208, 100:220]
        hsv_def = cv2.cvtColor(frame_def, cv2.COLOR_BGR2HSV)
        [masked_def, def_hue, _] = create_hue_mask(hsv_def, yellow_threshold[0:3], yellow_threshold[3:6])

        def_size.append(def_hue)
        cv2.imshow("Defender", masked_def) # todo remove runtime
        # Press q if you want to end the loop
        if cv2.waitKey(1) & 0xFF == ord('q'): # todo remove runtime
            frame_counter = 0
            break

    if frame_counter == 360 and defender_bool:
        set_point_def = np.average(sorted(def_size)[0:10])
    if frame_counter > 360 and defender_bool:
        if set_point_def - size_eps < def_hue < set_point_def + size_eps:
            client.write_single_register(4, 0)
            print("Defender calibrated")
            frame_counter = 0
            break

cv2.destroyAllWindows()

golie_bool = client.read_holding_registers(12, 1)
while golie_bool[0] == 0:
    golie_bool = client.read_holding_registers(12, 1)
    sleep(0.2)

print("Golie Wait")
sleep(2)
print("Golie Start")

while (nRet == ueye.IS_SUCCESS):
    frame_counter += 1
    array = ueye.get_data(pcImageMemory, width, height, nBitsPerPixel, pitch, copy=False)
    # ...reshape it in an numpy array...
    frame = np.reshape(array, (height.value, width.value, bytes_per_pixel))

    if golie_bool:
        frame_golie = frame[height.value - 220:height.value - 35, 300:450]
        hsv_golie = cv2.cvtColor(frame_golie, cv2.COLOR_BGR2HSV)
        [masked_golie, golie_hue, _] = create_hue_mask(hsv_golie, yellow_threshold[0:3], yellow_threshold[3:6])
        golie_size.append(golie_hue)

        cv2.imshow("Golie", masked_golie) #todo remove runtime
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if frame_counter == 360 and golie_bool:
        set_point_golie = np.average(sorted(golie_size)[0:10])

    if frame_counter > 360 and golie_bool:
        if set_point_golie - size_eps < golie_hue < set_point_golie + size_eps:
            client.write_single_register(5, 0)
            print("Golie calibrated")
            break

cv2.destroyAllWindows()

start_game = client.read_holding_registers(10,1)
while start_game[0] == 0:
    start_game = client.read_holding_registers(10,1)
    sleep(0.2)

client.write_multiple_registers(0, [600, 360, 525, 360])
print("The Game is on!")
sleep(2)
print("Starting main loop")
# ---------------------------------------------------------------------------------------------------------------------------------------
# Start main loop
total = np.zeros(1000)
i = 0
while (nRet == ueye.IS_SUCCESS):
    t0 = time.time()  #todo rm
    i += 1

    array = ueye.get_data(pcImageMemory, width, height, nBitsPerPixel, pitch, copy=False)
    # ...reshape it in an numpy array...
    frame = np.reshape(array, (height.value, width.value, bytes_per_pixel))
    # Image processing here
    frame_ball = frame[45:height.value - 70, 45:width.value - 75]

    hsv_ball = cv2.cvtColor(frame_ball, cv2.COLOR_BGR2HSV)

    [masked_ball, _, _] = create_hue_mask(hsv_ball, white_threshold[0:3], white_threshold[3:6])

    [top_left, template_found, centers_now] = detection_pattern(masked_ball)

    # cv2.imshow('game_mask', masked_ball)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

    x_pred = np.matmul(A, x_est)
    P_pred = np.matmul(np.matmul(A, P_est), np.transpose(A)) + Q

    if template_found:
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv2.rectangle(frame_ball, top_left, bottom_right, (128, 0, 0), 2)  # drawing of rectangle
        # y_ball = (centers_now[0] / 680)  # je to y v modelu
        # x_ball = (centers_now[1] / 485) # je to x v modelu

        #Kalman
        ball_coords = np.array([[centers_now[1]], [centers_now[0]]])
        K1 = np.matmul(np.matmul(C, P_pred), np.transpose(C)) + R
        K = np.matmul(P_pred, np.matmul(np.transpose(C), np.linalg.inv(K1)))
        x_est = x_pred + np.matmul(K, (ball_coords - np.matmul(C, x_pred)))
        P_est = P_pred - np.matmul(np.matmul(K, K1), np.transpose(K))
    else:
        x_est = x_pred
        P_est = P_pred

    # cv2.imshow('game', frame_ball)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

    if x_est[2] < 250:
        golie_lin = 130
    elif x_est[2] > 450:
        golie_lin = 900
    else:
        golie_lin = int((((x_est[2] - 250)/200) * 770) + 120) #golie slide linear transformation: 130-770

    if x_est[2] < 450:
        defender_lin = int(((x_est[2]/680) * 1275))
    else:
        defender_lin = int(((x_est[2]-270)/680) * 1275)

    if 130 > golie_lin > 900:
        golie_lin = 515
    if 10 > defender_lin > 1265:
        defender_lin = 600

    client.write_multiple_registers(0, [defender_lin, 360+random.randint(-40,40), golie_lin, 360+random.randint(-40,40)])

    if i <=999:
        t1 = time.time()
        total[i] = t1 - t0

# ---------------------------------------------------------------------------------------------------------------------------------------
# EXIT
print(np.average(total))
# Releases an image memory that was allocated using is_AllocImageMem() and removes it from the driver management
ueye.is_FreeImageMem(hCam, pcImageMemory, MemID)
# Disables the hCam camera handle and releases the data structures and memory areas taken up by the uEye camera
ueye.is_ExitCamera(hCam)

cv2.destroyAllWindows()

print()
print("END")
