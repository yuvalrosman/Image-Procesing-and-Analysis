import numpy as np
import matplotlib.pyplot as plt
import cv2


def max_freq_filtering(fshift, precentege):
    """
    Reconstruct an image using only its maximal amplitude frequencies.
    :param fshift: The fft of an image, **after fftshift** -
    complex float ndarray of size [H x W].
    :param precentege: the wanted precentege of maximal frequencies.
    :return:
    fMaxFreq: The filtered frequency domain result -
                        complex float ndarray of size [H x W].
    imgMaxFreq: The filtered image - real float ndarray of size [H x W].
    """
    amplitude = np.abs(fshift)
    threshhold = np.percentile(amplitude, 100 - precentege)
    mask_max = np.ones_like(fshift)
    mask_max[np.abs(fshift) < threshhold] = 0
    fMaxFreq = mask_max * fshift

    imgMaxFreq = np.abs(np.fft.ifft2(np.fft.ifftshift(fMaxFreq)))

    return fMaxFreq, imgMaxFreq


def interpolation(pixel_x, pixel_y, dx, dy, image):
    adj_mat = np.array([[image[pixel_y][pixel_x], image[pixel_y][(pixel_x + 1) % (image.shape[0])]]
                           , [image[(pixel_y + 1) % (image.shape[1])][pixel_x],
                              image[(pixel_y + 1) % (image.shape[1])][(pixel_x + 1) % (image.shape[0])]]])
    return np.array([1 - dx, dx]) @ adj_mat @ np.array([[dy], [1 - dy]])


def bilinear_displacement(dx, dy, image):
    """
    Calculate the displacement of a pixel using a bilinear interpolation.
    :param dx: the displacement in the x direction. dx in range [0,1).
    :param dy: the displacement in the y direction. dy in range [0,1).
    :param image: The image on which we preform the cyclic displacement
    :return:
            displaced_image: The new displaced image
    """
    displaced_image = np.ndarray(image.shape)

    for pixel, value in np.ndenumerate(image):
        displaced_image[pixel[1]][pixel[0]] = interpolation(pixel[1], pixel[0], dx, dy, image)
    return displaced_image


def general_displacement(dx, dy, image):
    """
    Calculate the displacement of a pixel using a bilinear interpolation.
    :param dx: the displacement in the x direction.
    :param dy: the displacement in the y direction.
    :param image: The image on which we preform the cyclic displacement
    :return:
            displaced_image: The new displaced image
    """
    # first displace by the whole number
    whole_displace = np.ndarray(image.shape)
    int_x = int(dx)
    int_y = int(dy)
    for pixel, value in np.ndenumerate(image):
        whole_displace[(pixel[1] + int_x) % (image.shape[1])][ (pixel[0] + int_y) % (image.shape[0])] = value
    displaced_image = bilinear_displacement(dx - int_x, dy - int_y, whole_displace)

    return displaced_image

def rotating_img(image, theta):
    """
    Calculate the displacement of a pixel using a bilinear interpolation.
    :param image: the image to rotate.
    :param theta: Angle of rotation in radians.
    :return:
        rotated_image: The new displaced image
    """
    center = [image.shape[0]/2,image.shape[1]/2]
    rotated_image = np.zeros_like(image)
    rotation_mat = np.array([[np.cos(theta), np.sin(theta)],[-np.sin(theta), np.cos(theta)]])
    for pixel,value in np.ndenumerate(image):
        pixel_from_center = [pixel[0] - center[0], pixel[1]-center[1]]
        new_pixel = np.array(pixel_from_center)@rotation_mat
        rounded_pixel = [round(new_pixel[0])+int(center[0]),round(new_pixel[1])+int(center[1])]
        if(0<=rounded_pixel[0]<image.shape[0] and 0<=rounded_pixel[1]<image.shape[1]):
            rotated_image[rounded_pixel[0],rounded_pixel[1]] =value
    return rotated_image





#1.a

building = cv2.imread("../my_data/building.jpg")
greyscale_building = cv2.cvtColor(building, cv2.COLOR_BGR2GRAY)
image = greyscale_building.astype(np.uint8)

cv2.imshow("Building",image)
cv2.waitKey(1000)




#1.b
image_fft = np.fft.fft2(image)
image_shifted_fft = np.fft.fftshift(image_fft)
fft_amplitude = np.log(1+np.abs(image_shifted_fft))
plt.imshow(fft_amplitude)
plt.title("Shifted FFT of Building")
plt.show()


#1.c

building_rows,building_cols = image_fft.shape
mask_k = np.ones_like(image_fft)
mask_l = np.ones_like(image_fft)
mask_cross =np.ones_like(image_fft)

low_2_precent_cols = int(0.02*building_cols)
low_2_precent_rows = int(0.02*building_rows)

#low_frequencies_cols = image_shifted_fft[:,:low_2_precent_cols]
mask_l[:,low_2_precent_cols:] = 0
mask_k[low_2_precent_rows:,:] = 0
mask_cross[low_2_precent_rows:,low_2_precent_cols:] = 0

masked_l_axis = image_fft*mask_l
masked_k_axis = image_fft*mask_k
masked_cross_axis = image_fft*mask_cross


#show the magnitude spectrum of the lowest 2 precent frequencies
fft_shift_cols = np.fft.fftshift(masked_l_axis)
fft_amplitude_cols = np.log(1+np.abs(fft_shift_cols))
plt.imshow(fft_amplitude_cols)
plt.title("Shifted FFT of Building 2 precent Cols")
plt.show()

fft_shift_rows = np.fft.fftshift(masked_k_axis)
fft_amplitude_rows = np.log(1+np.abs(fft_shift_rows))
plt.imshow(fft_amplitude_rows)
plt.title("Shifted FFT of Building 2 precent Rows")
plt.show()



fft_shift_cross = np.fft.fftshift(masked_cross_axis)
fft_amplitude_cross = np.log(1+np.abs(fft_shift_cross))
plt.imshow(fft_amplitude_cross)
plt.title("Shifted FFT of Building 2 precent Cross")
plt.show()

inversed_shift_cols = np.fft.ifftshift(fft_shift_cols)
inversed_image_cols = np.abs(np.fft.ifft2(inversed_shift_cols))
plt.imshow(inversed_image_cols,cmap = "gray")
plt.title("image of Building 2 precent Cols")
plt.show()

inversed_shift_rows = np.fft.ifftshift(fft_shift_rows)
inversed_image_rows = np.abs(np.fft.ifft2(inversed_shift_rows))
plt.imshow(inversed_image_rows,cmap = "gray")
plt.title("image of Building 2 precent Rows")
plt.show()

inversed_shift_cross = np.fft.ifftshift(fft_shift_cross)
inversed_image_cross = np.abs(np.fft.ifft2(inversed_shift_cross))
plt.imshow(inversed_image_cross,cmap = "gray")
plt.title("image of Building 2 precent Cross")
plt.show()

#1.d
max_filter_amp_10,max_filter_image_10 =max_freq_filtering(image_shifted_fft,10)

plt.imshow(np.log(1+np.abs(max_filter_amp_10)))
plt.title("max amps")
plt.show()

plt.imshow(max_filter_image_10,cmap = "gray")
plt.title("max amp image")
plt.show()

#1.e
max_filter_amp_4,max_filter_image_4 =max_freq_filtering(image_shifted_fft,4)

plt.imshow(np.log(1+np.abs(max_filter_amp_4)))
plt.title("max amps")
plt.show()

plt.imshow(max_filter_image_4,cmap = "gray")
plt.title("max amp image")
plt.show()


#1.f
graph =[]
rows,cols = image_shifted_fft.shape
for i in range(1,100):
    curr_img = max_freq_filtering(image_shifted_fft, i)[1]
    graph.append(np.sum((image-curr_img)**2)/(rows*cols))
plt.plot(list(range(1, 100)), graph)
plt.show()




#Question 2
#2.a
parrot = cv2.imread("../given_data/parrot.png")
selfie = cv2.imread("../my_data/yours.jpg")
width,height = parrot.shape[:2]
resize_selfie=cv2.resize(selfie,(width,height))
greyscale_selfie = cv2.cvtColor(resize_selfie, cv2.COLOR_BGR2GRAY)
greyscale_parrot = cv2.cvtColor(parrot, cv2.COLOR_BGR2GRAY)

print(selfie.shape)
selfie_image = greyscale_selfie.astype(np.uint8)
parrot_image = greyscale_parrot.astype(np.uint8)


plt.imshow(parrot_image, cmap="gray")
plt.title("parrot")
plt.show()

plt.imshow(selfie_image, cmap="gray")
plt.title("selfie")
plt.show()


#2.b

parrot_fft = np.fft.fft2(parrot_image)
parrot_shifted_fft = np.fft.fftshift(parrot_fft)

selfie_fft = np.fft.fft2(selfie_image)
selfie_shifted_fft = np.fft.fftshift(selfie_fft)

phase_parrot=np.angle(parrot_shifted_fft)
amp_parrot=np.abs(parrot_shifted_fft)

phase_selfie=np.angle(selfie_shifted_fft)
amp_selfie=np.abs(selfie_shifted_fft)

plt.imshow(amp_parrot, cmap="gray")
plt.title("Parrot Amp")
plt.show()

plt.imshow(amp_selfie, cmap="gray")
plt.title("Selfie Amp")
plt.show()

plt.imshow(phase_parrot, cmap="gray")
plt.title("Parrot Phase")
plt.show()

plt.imshow(phase_selfie, cmap="gray")
plt.title("Selfie Phase")
plt.show()


#2.c

selfie_amp_parrot_phase_fft = amp_selfie*np.exp(1j*phase_parrot)
parrot_amp_selfie_phase_fft = amp_parrot*np.exp(1j*phase_selfie)

selfie_amp_parrot_phase = np.abs(np.fft.ifft2(np.fft.ifftshift(selfie_amp_parrot_phase_fft)))
parrot_amp_selfie_phase = np.abs(np.fft.ifft2(np.fft.ifftshift(parrot_amp_selfie_phase_fft)))

plt.imshow(selfie_amp_parrot_phase, cmap="gray")
plt.title("Selfie Amp Parrot Phase")
plt.show()

plt.imshow(parrot_amp_selfie_phase, cmap="gray")
plt.title("Parrot Amp Selfie Phase")
plt.show()

min_amp,max_amp= np.min(amp_selfie), np.max(amp_selfie)
selfie_amp_random_phase = amp_selfie*np.exp(1j*np.random.uniform(size=selfie_shifted_fft.shape,low=0,high=2*np.pi))
selfie_phase_random_amp = np.random.uniform(size=selfie_shifted_fft.shape,low=min_amp,high=max_amp) * np.exp(1j*phase_selfie)

selfie_amp_random_phase = np.abs(np.fft.ifft2(np.fft.ifftshift(selfie_amp_random_phase)))
random_amp_selfie_phase = np.abs(np.fft.ifft2(np.fft.ifftshift(selfie_phase_random_amp)))

plt.imshow(selfie_amp_random_phase, cmap="gray")
plt.title("Selfie Amp Random Phase")
plt.show()

plt.imshow(random_amp_selfie_phase, cmap="gray")
plt.title("selfie_phase_random_amp")
plt.show()



#Question 3
#3.b
grid_x, grid_y = np.meshgrid(np.linspace(0,511,512),np.linspace(0,511,512))
f1_image  = np.sin(grid_x*2*np.pi*5/512) +  np.sin(grid_y*2*np.pi*40/512) +np.sin((grid_x+grid_y)*2*np.pi*2/512)


plt.imshow(f1_image, cmap="gray")
plt.title("F1 Space Domain")
plt.show()

f1_image_fft = np.fft.fft2(f1_image)
f1_image_shifted_fft = np.fft.fftshift(f1_image_fft)
f1_image_shifted_fft_amp = np.log(1+np.abs(f1_image_shifted_fft))

plt.imshow(f1_image_shifted_fft_amp)
plt.title("F1 FFT Frequency Domain")
plt.show()


#3.c
grid_x_10, grid_y_10 = np.meshgrid(np.linspace(0,511,52),np.linspace(0,511,52))
f10_image  = np.sin(grid_x_10*2*np.pi*5/512) +  np.sin(grid_y_10*2*np.pi*40/512) +np.sin((grid_x_10+grid_y_10)*2*np.pi*2/512)

plt.imshow(f10_image, cmap="gray")
plt.title("F10 Space Domain")
plt.show()

f10_image_fft = np.fft.fft2(f10_image)
f10_image_shifted_fft = np.fft.fftshift(f10_image_fft)
f10_image_shifted_fft_amp = np.log(1+np.abs(f10_image_shifted_fft))

plt.imshow(f10_image_shifted_fft_amp)
plt.title("F10 FFT Frequency Domain")
plt.show()

#3.e

mandrill_raw = cv2.imread("../given_data/Mandrill.jpg")
greyscale_mandrill = cv2.cvtColor(mandrill_raw, cv2.COLOR_BGR2GRAY)
mandrill = greyscale_mandrill.astype(np.uint8)

plt.imshow(mandrill, cmap ="gray")
plt.title("Mandrill")
plt.show()


mandrill_image_fft = np.fft.fft2(mandrill)
mandrill_image_shifted_fft = np.fft.fftshift(mandrill_image_fft)
mandrill_image_shifted_fft_amp = np.log(1+np.abs(mandrill_image_shifted_fft))

plt.imshow(mandrill_image_shifted_fft_amp)
plt.title("Mandrill FFT Frequency Domain")
plt.show()

#3.f

downsampled_mandrill = mandrill[::4,::4]
plt.imshow(downsampled_mandrill, cmap ="gray")
plt.title("Downsampled Mandrill")
plt.show()


downsampled_mandrill_image_fft = np.fft.fft2(downsampled_mandrill)
downsampled_mandrill_image_shifted_fft = np.fft.fftshift(downsampled_mandrill_image_fft)
downsampled_mandrill_image_shifted_fft_amp = np.log(1+np.abs(downsampled_mandrill_image_shifted_fft))

plt.imshow(downsampled_mandrill_image_shifted_fft_amp)
plt.title("Downsamples Mandrill FFT Frequency Domain")
plt.show()








# Question 4
# 4.a +4.b functions

# 4.c

cameraman = cv2.imread("../given_data/cameraman.jpg")
greyscale_cameraman = cv2.cvtColor(cameraman, cv2.COLOR_BGR2GRAY)
cameraman_image = greyscale_cameraman.astype(np.uint8)

displaced_cameraman = general_displacement(150.7, 110.4, cameraman_image)

plt.imshow(displaced_cameraman, cmap="gray")
plt.title("cameraman displaced")
plt.show()

#4.d
ryan_raw = cv2.imread("../given_data/Ryan.jpg")
greyscalse_ryan = cv2.cvtColor(ryan_raw, cv2.COLOR_BGR2GRAY)
ryan = greyscalse_ryan.astype(np.uint8)

mask1 = np.zeros_like(ryan)
ryan_face_center_y, ryan_face_center_x = 45,375 #trail and error
for (y,x), value in np.ndenumerate(ryan):
    if((y - ryan_face_center_y) ** 2) + ((x - ryan_face_center_x) ** 2) <= 155 ** 2 and y >= ryan_face_center_y:
        mask1[y,x] = 1

ryan_win = mask1*ryan

plt.imshow(ryan_win, cmap="gray")
plt.title("ryan win!")
plt.show()


#4.e
ryan_pi_half = rotating_img(ryan_win,np.pi/2)
plt.imshow(ryan_pi_half, cmap="gray")
plt.title("ryan win pi/2!")
plt.show()

ryan_pi_third = rotating_img(ryan_win,np.pi/3)
plt.imshow(ryan_pi_third, cmap="gray")
plt.title("ryan win pi/3!")
plt.show()


ryan_pi_quarter = rotating_img(ryan_win,np.pi/4)
plt.imshow(ryan_pi_quarter, cmap="gray")
plt.title("ryan win pi/4!")
plt.show()


