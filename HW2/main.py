import numpy
import numpy as np
import matplotlib.pyplot as plt
import cv2


def gamma_correction(img, gamma): #Q1c
    """
    Perform gamma correction on a grayscale image.
    :param img: An input grayscale image - ndarray of uint8 type.
    :param gamma: the gamma parameter for the correction.
    :return:
    gamma_img: An output grayscale image after gamma correction -
    uint8 ndarray of size [H x W x 1].
    """
    gamma_img = np.uint8(((img / 255.0) ** gamma) * 255.0)
    return gamma_img



def video_to_frames(vid_path: str, start_second, end_second): #Q2a
    """
    Load a video and return its frames from the wanted time range.
    :param vid_path: video file path.
    :param start_second: time of first frame to be taken from the
    video in seconds.
    :param end_second: time of last frame to be taken from the
    video in seconds.
    :return:
    frame_set: a 4D uint8 np array of size [num_of_frames x H x W x C]
    containing the wanted video frames.
    """
    capture = cv2.VideoCapture(vid_path)
    frames =[]
    start_frame = int(start_second * capture.get(cv2.CAP_PROP_FPS)) if start_second is not None else 0
    end_frame = int(end_second * capture.get(cv2.CAP_PROP_FPS)) if end_second is not None else int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    if start_second == end_second:
        ret, frame = capture.read()
        resized_frame = cv2.resize(frame, (frame.shape[1], frame.shape[0]))
        frames.append(resized_frame)
        frame_set = np.array(frames)
        capture.release()
        return frame_set

    while capture.get(cv2.CAP_PROP_POS_FRAMES) < end_frame:
        # Read a single frame from the video
        ret, frame = capture.read()

        # If no frame is read, break the loop
        if not ret:
            break
        resized_frame = cv2.resize(frame, (frame.shape[1], frame.shape[0]))

        # Append the resized frame to the list of frames
        frames.append(resized_frame)

        # Convert the list of frames to a NumPy array
    frame_set = np.array(frames)

    # Release the video capture object
    capture.release()

    return frame_set


def match_corr(corr_obj, img): #Q2b
    """
    return the center coordinates of the location of 'corr_obj' in 'img'.
    :param corr_obj: 2D numpy array of size [H_obj x W_obj]
    containing an image of a component.
    :param img: 2D numpy array of size [H_img x W_img]
    where H_img >= H_obj and W_img>=W_obj,
    containing an image with the 'corr_obj' component in it.
    :return:
    match_coord: the two center coordinates in 'img'
    of the 'corr_obj' component.
    """
    cross_corr = cv2.filter2D(img, 4, corr_obj, borderType=cv2.BORDER_CONSTANT) #preform the correlation
    auto_corr = cv2.filter2D(corr_obj, 4, corr_obj, borderType=cv2.BORDER_CONSTANT)  # preform the correlation
    distance = np.abs(cross_corr - np.max(auto_corr))
    match_coord =np.unravel_index(np.argmin(distance),distance.shape)
    return match_coord


def poisson_noisy_image(X, a): ##Q4a
    """
    Creates a Poisson noisy image.
    :param X: The Original image. np array of size [H x W] and of type uint8.
    :param a: number of photons scalar factor
    :return:
    Y: The noisy image. np array of size [H x W] and of type uint8.
    """
    float_image = np.array(X, dtype=np.float)
    num_of_photons = a*float_image
    photons_noisy = np.random.poisson(num_of_photons)
    frame_noisy = photons_noisy/a
    clipped_frame_noisy = np.clip(frame_noisy,0,255)
    Y = np.array(clipped_frame_noisy, dtype=np.uint8)
    return Y

def Gk(lambda_reg,kernal,Xk_column,Y_column,X_size):
    Xk_as_matrix = np.reshape(Xk_column,X_size,'F')
    Gk_as_matrix = cv2.filter2D(Xk_as_matrix,-1,kernal)
    Gk_as_matrix = cv2.filter2D(Gk_as_matrix, -1, kernal)
    return lambda_reg*(Gk_as_matrix).flatten('F') + Xk_column - Y_column      ## the matrix trasposed is the same since its symetrical

def Miu(Gk,kernal,lambda_reg,X_size):
    Gk_as_matrix = np.reshape(Gk,X_size,'F')
    Gk_with_kernal = cv2.filter2D(Gk_as_matrix,-1,kernal)
    Gk_with_kernal = cv2.filter2D(Gk_with_kernal, -1, kernal)
    Gk_with_kernal_flatten = Gk_with_kernal.flatten('F')
    Gk_transpose = np.transpose(Gk)
    miu = (Gk_transpose@Gk)/(Gk_transpose@Gk + lambda_reg*Gk_transpose@Gk_with_kernal_flatten)
    return miu.flatten('F')

def Err1calc(Xk_column,Y_column,lambda_reg,kernal,X_size):
    Xk_as_matrix = np.reshape(Xk_column, X_size, 'F')
    Xk_with_kernal = cv2.filter2D(Xk_as_matrix,-1,kernal)
    Xk_with_kernal_flatten = Xk_with_kernal.flatten('F')
    first_exp = (np.transpose(Xk_column-Y_column))@(Xk_column-Y_column)
    second_exp = lambda_reg * (np.transpose(Xk_with_kernal_flatten)@(Xk_with_kernal_flatten))
    return first_exp + second_exp

def denoise_by_l2(Y, X, num_iter, lambda_reg): ##Q4b
    """
    L2 image denoising.
    :param Y: The noisy image. np array of size [H x W]
    :param X: The Original image. np array of size [H x W]
    :param num_iter: the number of iterations for the algorithm perform
    :param lambda_reg: the regularization parameter
    :return:
    Xout: The restored image. np array of size [H x W]
    Err1: The error between Xk at every iteration and Y.
    np array of size [num_iter]
    Err2: The error between Xk at every iteration and X.
    np array of size [num_iter]
    """
    D = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])
    X_column = X.flatten('F').astype(float)
    Y_column = Y.flatten('F').astype(float)
    Xk_column = Y_column.copy()
    Err1,Err2 = np.zeros(num_iter),np.zeros(num_iter)
    for iter in range (num_iter):
        Gk_column = Gk(lambda_reg,D,Xk_column,Y_column,X.shape)
        Miuk = Miu(Gk_column,D,lambda_reg,X.shape)
        Xk_column = Xk_column -Miuk*Gk_column
        Err1[iter] = Err1calc(Xk_column, Y_column, lambda_reg, D, X.shape)
        Err2[iter] = (np.transpose(Xk_column - X_column)) @ (Xk_column - X_column)
    Xout = np.reshape(Xk_column,X.shape,'F')
    return Xout, Err1, Err2




#Q1
#1.a

puppy = cv2.imread("../given_data/puppy.jpg")
greyscale_puppy = cv2.cvtColor(puppy, cv2.COLOR_BGR2GRAY)
puppy_image = greyscale_puppy.astype(np.uint8)

plt.imshow(puppy_image, cmap = "gray")
plt.title("Puppy Greyscale")
plt.show()

#1.b

histogram = cv2.calcHist([puppy_image], [0], None, [256], [0, 256])

plt.plot(histogram)
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.title('Puppy Histogram')
plt.show()

#1.c

gamma_bright_puppy = gamma_correction(puppy_image,0.5)
gamma_dark_puppy = gamma_correction(puppy_image,1.5)


plt.imshow(gamma_bright_puppy, cmap = "gray")
plt.title("Puppy Bright Gamma Correction")
plt.show()

histogram_bright = cv2.calcHist([gamma_bright_puppy], [0], None, [256], [0, 256])

plt.plot(histogram_bright)
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.title('Puppy Bright Histogram')
plt.show()



plt.imshow(gamma_dark_puppy, cmap = "gray")
plt.title("Puppy Dark Gamma Correction")
plt.show()


histogram_dark = cv2.calcHist([gamma_dark_puppy], [0], None, [256], [0, 256])

plt.plot(histogram_dark)
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.title('Puppy Dark Histogram')
plt.show()






#Q2

#2.a done

#2.b done

#2.c

frames = video_to_frames("../given_data/Corsica.mp4",250,260)
frames_greyscale = np.stack([(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)) for frame in frames])
height, width = frames_greyscale.shape[1], frames_greyscale.shape[2]
bottom_start = height // 3
bottom_end = height

# Extract the bottom two-thirds of the image
processed = frames_greyscale[:, bottom_start:bottom_end, 7:627]
processed_final = np.stack([frame for frame in processed])
processed_height, processed_width = processed_final.shape[1],processed_final.shape[2]
#2.d

panorama_base = np.zeros((processed_height, int(2.5*processed_width)))
mid_frame,early_frame,late_frame = processed_final[125],processed_final[50],processed_final[200]


start_row = (panorama_base.shape[0] - mid_frame.shape[0]) // 2
start_col = (panorama_base.shape[1] - mid_frame.shape[1]) // 2

# Calculate the ending indices for the image
end_row = start_row + mid_frame.shape[0]
end_col = start_col + mid_frame.shape[1]

# Place the image at the calculated indices
panorama_base[start_row:end_row, start_col:end_col] = mid_frame

plt.imshow(panorama_base, cmap = "gray")
plt.title("Panorama Base")
plt.show()

plt.imshow(early_frame, cmap = "gray")
plt.title("Early Frame")
plt.show()

plt.imshow(late_frame, cmap = "gray")
plt.title("Late Frame")
plt.show()

#2.e

sub_early_rectangle = early_frame[:,0:150]
sub_late_rectangle = late_frame[:,470:620]


right_match_coord = match_corr(sub_early_rectangle,panorama_base)
left_match_coord = match_corr(sub_late_rectangle,panorama_base)

plt.imshow(sub_early_rectangle, cmap = "gray")
plt.title(right_match_coord)
plt.show()

plt.imshow(sub_late_rectangle, cmap = "gray")
plt.title(left_match_coord)
plt.show()


#2.f

panorama = panorama_base.copy()
start_of_early_position = (right_match_coord[1]-75,max(0,right_match_coord[0]-120))
end_of_early_position = (min(1550,right_match_coord[1]+545),min(240,right_match_coord[0]+120))

start_of_late_position = (max(0,left_match_coord[1]-545),max(0,left_match_coord[0]-120))
end_of_late_position = (min(1550,left_match_coord[1]+75),min(240,left_match_coord[0]+120))

size_of_early = (end_of_early_position[0]-start_of_early_position[0],end_of_early_position[1]-start_of_early_position[1])
size_of_late  = (end_of_late_position[0]-start_of_late_position[0],end_of_late_position[1]-start_of_late_position[1])

avg_of_early = np.zeros_like(early_frame)
avg_of_early[0:240,0:150] = (early_frame[0:240,0:150] + panorama[start_of_early_position[1]:start_of_early_position[1]+240,start_of_early_position[0]:start_of_early_position[0]+150])/2

avg_of_late = np.zeros_like(early_frame)

avg_of_late[0:240,0:150] = (late_frame[0:240,470:620] + panorama[start_of_late_position[1]:start_of_early_position[1]+240,end_of_late_position[0]-150:end_of_late_position[0]])/2


panorama[start_of_early_position[1]:end_of_early_position[1],start_of_early_position[0]:end_of_early_position[0]] =  early_frame[0:size_of_early[1],0:size_of_early[0]]
panorama[start_of_late_position[1]:end_of_late_position[1],start_of_late_position[0]:end_of_late_position[0]] = late_frame[0:size_of_late[1],0:size_of_late[0]]
panorama[start_of_early_position[1]:start_of_early_position[1]+240,start_of_early_position[0]:start_of_early_position[0]+150] = avg_of_early[0:240,0:150]
panorama[start_of_late_position[1]:start_of_late_position[1]+240,end_of_late_position[0]-150:end_of_late_position[0]] = avg_of_late[0:240,0:150]


plt.imshow(panorama, cmap = "gray")
plt.title("Panorama Final Result")
plt.show()

panorama_with_base_on_top = panorama.copy()
panorama_with_base_on_top[0:240, 465:1085] = mid_frame

plt.imshow(panorama_with_base_on_top, cmap = "gray")
plt.title("Panorama With Base On Top")
plt.show()




#Q3
#3.a
keyboard = cv2.imread("../given_data/keyboard.jpg")
greyscale_keyboard = cv2.cvtColor(keyboard, cv2.COLOR_BGR2GRAY)
keyboard_image = greyscale_keyboard.astype(np.uint8)

vertical_kernel = np.zeros((8, 1), dtype=np.uint8)
vertical_kernel[:, 0] = 1

horizontal_kernel = np.zeros((1, 8), dtype=np.uint8)
horizontal_kernel[0, :] = 1

horizontal_erosion = cv2.erode(keyboard_image,horizontal_kernel,iterations=1)
vertical_erosion = cv2.erode(keyboard_image,vertical_kernel,iterations=1)

plt.imshow(keyboard_image, cmap = "gray")
plt.title("Keyboard")
plt.show()


plt.imshow(horizontal_erosion, cmap = "gray")
plt.title("Horizontal Erosion")
plt.show()

plt.imshow(vertical_erosion, cmap = "gray")
plt.title("Vertical Erosion")
plt.show()

sum = horizontal_erosion+ vertical_erosion

plt.imshow(sum, cmap = "gray")
plt.title("Sum Of Images")
plt.show()


threshold_val = int(0.2*255)
binary_image = np.where(sum>=threshold_val,1,0)

plt.imshow(binary_image, cmap = "gray")
plt.title("Binary Image")
plt.show()


#3.b

inverse_binary_image = np.array(np.where(binary_image==1,0,1),dtype=np.uint8)

plt.imshow(inverse_binary_image, cmap = "gray")
plt.title("Inverse Binary Image")
plt.show()

keyboard_median_filtered = cv2.medianBlur(inverse_binary_image,9)

plt.imshow(keyboard_median_filtered, cmap = "gray")
plt.title("Median Filtered Image")
plt.show()

#3.c

square_kernel = np.ones((8, 8), dtype=np.uint8)
meadian_erosion = cv2.erode(keyboard_median_filtered,square_kernel,iterations=1)

plt.imshow(meadian_erosion, cmap = "gray")
plt.title("Median Erosion Image")
plt.show()

#3.d

multiplication_keyboard = keyboard_image*meadian_erosion

plt.imshow(multiplication_keyboard, cmap = "gray")
plt.title("Multiplication Keyboard Image")
plt.show()

K = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])

filtered_keyboard = cv2.filter2D(multiplication_keyboard,-1,K)

plt.imshow(filtered_keyboard, cmap = "gray")
plt.title("Filtered Keyboard Image")
plt.show()


##finding the best threshold

for i in range(1,4):
    curr_threshold = int(0.25*i*255)
    binary_sharpened_image = np.array(np.where(filtered_keyboard>=curr_threshold,1,0),dtype=np.uint8)
    plt.imshow(binary_sharpened_image, cmap="gray")
    plt.title("Binary Sharpened Keyboard Image With " + str(i*25) +"% Threshold")
    plt.show()




#Q4
#4.a

frames = video_to_frames("../given_data/Flash Gordon Trailer.mp4",20,21)
our_frame =frames[4] ##we chose frame 4
plt.imshow(our_frame)
plt.title("Frame")
plt.show()

green_frame = our_frame[:,:,1]

plt.imshow(green_frame, cmap='gray')
plt.title("Green Frame")
plt.show()

resized_image = cv2.resize(green_frame,(int(green_frame.shape[1]*0.5),int(green_frame.shape[0]*0.5)))

noisy_image = poisson_noisy_image(resized_image,3)

plt.imshow(noisy_image, cmap='gray')
plt.title("Noisy Image")
plt.show()

#4.b

denoised = denoise_by_l2(noisy_image,resized_image,50,0.5)
plt.imshow(denoised[0], cmap='gray')
plt.title("DeNoisy Image")
plt.show()


iterations = list(range(50))
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(iterations, np.log(denoised[1]), label='Err1')
ax.plot(iterations, np.log(denoised[2]), label='Err2')
ax.legend()
plt.title('Errors per iteration')
plt.grid()
plt.show()

#4.c

frames = video_to_frames("../given_data/Flash Gordon Trailer.mp4",38,39)
our_frame =frames[18] ##we chose frame 4
plt.imshow(our_frame)
plt.title("Frame")
plt.show()

green_frame = our_frame[:,:,1]

plt.imshow(green_frame, cmap='gray')
plt.title("Green Frame")
plt.show()

resized_image = cv2.resize(green_frame,(int(green_frame.shape[1]*0.5),int(green_frame.shape[0]*0.5)))

noisy_image = poisson_noisy_image(resized_image,3)

plt.imshow(noisy_image, cmap='gray')
plt.title("Noisy Image")
plt.show()

denoised = denoise_by_l2(noisy_image,resized_image,50,0.5)
plt.imshow(denoised[0], cmap='gray')
plt.title("DeNoisy Image")
plt.show()


iterations = list(range(50))
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(iterations, np.log(denoised[1]), label='Err1')
ax.plot(iterations, np.log(denoised[2]), label='Err2')
ax.legend()
plt.title('Errors per iteration')
plt.grid()
plt.show()



