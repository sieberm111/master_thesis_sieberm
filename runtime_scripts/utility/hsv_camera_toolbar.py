#Libraries
from pyueye import ueye
import numpy as np
import cv2
import sys


# setup
setup = [150, 100, 100, 180, 255, 255, 100, 500, 100, 400]  # [min H, min S, min V, max H, max S, max V]
setup_white = [0, 0, 125, 100, 150, 255]  # [min H, min S, min V, max H, max S, max V]
setup_yellow = [0, 255, 130, 255, 255, 255]  # [min H, min S, min V, max H, max S, max V]
setup_pink = [150, 100, 0, 210, 255, 255]  # [min H, min S, min V, max H, max S, max V]

max_value = 255
max_value_H = 360 // 2
low_H = 0
low_S = 0
low_V = 0
high_H = max_value_H
high_S = max_value
high_V = max_value
window_capture_name = 'Video Capture'
window_detection_name = 'Object Detection'
low_H_name = 'Low H'
low_S_name = 'Low S'
low_V_name = 'Low V'
high_H_name = 'High H'
high_S_name = 'High S'
high_V_name = 'High V'


def on_low_H_thresh_trackbar(val):
    global low_H
    global high_H
    low_H = val
    low_H = min(high_H - 1, low_H)
    cv2.setTrackbarPos(low_H_name, window_detection_name, low_H)

#
# def on_high_H_thresh_trackbar(val):
#     global low_H
#     global high_H
#     high_H = val
#     high_H = max(high_H, low_H + 1)
#     cv2.setTrackbarPos(high_H_name, window_detection_name, high_H)
#
#
# def on_low_S_thresh_trackbar(val):
#     global low_S
#     global high_S
#     low_S = val
#     low_S = min(high_S - 1, low_S)
#     cv2.setTrackbarPos(low_S_name, window_detection_name, low_S)
#
#
# def on_high_S_thresh_trackbar(val):
#     global low_S
#     global high_S
#     high_S = val
#     high_S = max(high_S, low_S + 1)
#     cv2.setTrackbarPos(high_S_name, window_detection_name, high_S)
#
#
# def on_low_V_thresh_trackbar(val):
#     global low_V
#     global high_V
#     low_V = val
#     low_V = min(high_V - 1, low_V)
#     cv2.setTrackbarPos(low_V_name, window_detection_name, low_V)
#
#
# def on_high_V_thresh_trackbar(val):
#     global low_V
#     global high_V
#     high_V = val
#     high_V = max(high_V, low_V + 1)
#     cv2.setTrackbarPos(high_V_name, window_detection_name, high_V)


def create_hue_mask(image, lower_color, upper_color):
    lower = np.array(lower_color, np.uint8)
    upper = np.array(upper_color, np.uint8)

    # Create a mask from the colors
    mask = cv2.inRange(image, lower, upper)
    output_image = cv2.bitwise_and(image, image, mask=mask)
    return mask
#---------------------------------------------------------------------------------------------------------------------------------------

#Variables
hCam = ueye.HIDS(0)             #0: first available camera;  1-254: The camera with the specified camera ID

# test set params from ini file
# pParam = ueye.wchar_p()
# pParam.value = "ids_camera2.ini"
# ueye.is_ParameterSet(hCam, ueye.IS_PARAMETERSET_CMD_LOAD_FILE, pParam, 0)


sInfo = ueye.SENSORINFO()
cInfo = ueye.CAMINFO()
pcImageMemory = ueye.c_mem_p()
MemID = ueye.int()
rectAOI = ueye.IS_RECT()
pitch = ueye.INT()
nBitsPerPixel = ueye.INT(24)    #24: bits per pixel for color mode; take 8 bits per pixel for monochrome
channels = 3                    #3: channels for color mode(RGB); take 1 channel for monochrome
m_nColorMode = ueye.INT()		# Y8/RGB16/RGB24/REG32
bytes_per_pixel = int(nBitsPerPixel / 8)
#---------------------------------------------------------------------------------------------------------------------------------------
print("START")
print()

# Starts the driver and establishes the connection to the camera
nRet = ueye.is_InitCamera(hCam, None)
if nRet != ueye.IS_SUCCESS:
    print("is_InitCamera ERROR")

#test set params from ini file
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

#nRet = ueye.is_ResetToDefault( hCam)
#if nRet != ueye.IS_SUCCESS:
#    print("is_ResetToDefault ERROR")

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

#---------------------------------------------------------------------------------------------------------------------------------------

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


# spawn_controllers()

cv2.namedWindow(window_detection_name)
cv2.createTrackbar(low_H_name, window_detection_name, low_H, max_value_H, on_low_H_thresh_trackbar)
# cv2.createTrackbar(high_H_name, window_detection_name, high_H, max_value_H, on_high_H_thresh_trackbar)
# cv2.createTrackbar(low_S_name, window_detection_name, low_S, max_value, on_low_S_thresh_trackbar)
# cv2.createTrackbar(high_S_name, window_detection_name, high_S, max_value, on_high_S_thresh_trackbar)
# cv2.createTrackbar(low_V_name, window_detection_name, low_V, max_value, on_low_V_thresh_trackbar)
# cv2.createTrackbar(high_V_name, window_detection_name, high_V, max_value, on_high_V_thresh_trackbar)


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

#---------------------------------------------------------------------------------------------------------------------------------------

# Continuous image display
while(nRet == ueye.IS_SUCCESS):
    # In order to display the image in an OpenCV window we need to...
    # ...extract the data of our image memory
    array = ueye.get_data(pcImageMemory, width, height, nBitsPerPixel, pitch, copy=False)

    # bytes_per_pixel = int(nBitsPerPixel / 8)

    # ...reshape it in an numpy array...
    frame = np.reshape(array, (height.value, width.value, bytes_per_pixel))

#---------------------------------------------------------------------------------------------------------------------------------------
    #Include image data processing here

    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    white_hue = create_hue_mask(hsv_image, [low_H, low_S, low_V], [high_H, high_S, high_V])
    # frame1 = cv2.cvtColor(white_hue, cv2.COLOR_HSV2BGR)
#---------------------------------------------------------------------------------------------------------------------------------------
    cv2.imshow(window_detection_name, white_hue)
    #...and finally display it

    # Press q if you want to end the loop
    if cv2.waitKey() & 0xFF == ord('q'):
        new_setup = [low_H, low_S, low_V, high_H, high_S, high_V]
        print("new_setup = {}", new_setup)


        break
#---------------------------------------------------------------------------------------------------------------------------------------

# Releases an image memory that was allocated using is_AllocImageMem() and removes it from the driver management
ueye.is_FreeImageMem(hCam, pcImageMemory, MemID)

# Disables the hCam camera handle and releases the data structures and memory areas taken up by the uEye camera
ueye.is_ExitCamera(hCam)

# Destroys the OpenCv windows
cv2.destroyAllWindows()

# frame set template

print()
print("END")
