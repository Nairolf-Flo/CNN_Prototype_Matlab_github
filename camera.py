from picamera import PiCamera
from time import sleep

camera = PiCamera()
camera.rotation = 180
camera.resolution = (64, 64)
camera.exposure_mode = 'nightpreview'

camera.capture("photo.jpg")