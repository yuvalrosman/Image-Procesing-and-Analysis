# imports for the HW
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import torch
import torchvision
import random

def max_lloyd_quantizer(data, levels, meps):
    """
    The function implements the iterative Max-Llyod algorithm for
    image quantization, and return the quantized image and some
    parameters of the quantization.
    inputs:
    data: one channel image in a uint8 format.
    levels: number of wanted different representation levels.
    meps: minimal required approximation.
    outpus:
    dataout: the image after the quantization.
    distortion: a vector with the size 1 X number of
    iterations. The vector contains the
    average distortion of the quantized
    image in each iteration.
    QL: a vector with the length of levels that contains
    the different representation levels.
    """
    # ====== YOUR CODE: ======
    QL = [0] * levels
    r_vector = np.sort(np.random.choice(np.ravel(data),levels+1,replace=False))
    hist, bin_num = np.histogram(data.flatten(), bins=256)
    pdf = hist/data.size
    distortion =[] # distortion vector
    dataout = np.zeros_like(data)
    iter_num = 50
    m=0
    while(m<iter_num):
        for k in range (1,levels+1):
            sum1 = sum(u*pdf[u] for u in range(int(r_vector[k-1]),int(r_vector[k])))
            sum2 = sum(pdf[u] for u in range(int(r_vector[k - 1]), int(r_vector[k])))
            if sum1 != 0:
                QL[k - 1] = sum1/sum2
        for k in range(levels-1):
            r_vector[k] = (QL[k]+QL[k+1])/2
        distortion.append(np.mean((data-dataout)**2,dtype =np.float64))
        for i in range(levels):
            dataout[(r_vector[i] <= data) & (data < r_vector[i+1])] = QL[i]
        dataout[data == r_vector[levels]] = QL[levels-1]
        if m>0:
            epsilon = (np.abs(distortion[m] - distortion[m-1]))/distortion[m-1]
            if epsilon < meps:
                break
        m+=1
    # ========================
    return dataout, distortion, QL




#1.a
#Done

#1.b
image = cv2.imread("../given_data/colorful.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_float = image.astype(float)


figure = plt.figure(figsize=(10,10))
ax = figure.add_subplot (1,1,1)
ax.imshow(image_rgb)
ax.set_title ("colorful in RGB")
plt.show()

meps = 0.01

dataout_red_6, distortion_red_6, QL_red_6 = max_lloyd_quantizer(image_float[:,:,0], 6, meps)
dataout_green_6, distortion_green_6, QL_green_6 = max_lloyd_quantizer(image_float[:,:,1], 6, meps)
dataout_blue_6, distortion_blue_6, QL_blue_6 = max_lloyd_quantizer(image_float[:,:,2], 6, meps)
dataout_red_15, distortion_red_15, QL_red_15 = max_lloyd_quantizer(image_float[:,:,0], 15, meps)
dataout_green_15, distortion_green_15, QL_green_15 = max_lloyd_quantizer(image_float[:,:,1], 15, meps)
dataout_blue_15, distortion_blue_15, QL_blue_15 = max_lloyd_quantizer(image_float[:,:,2], 15, meps)

figure = plt.figure(figsize=(10,10))
ax = figure.add_subplot (1,1,1)
ax.imshow(dataout_red_6)
ax.set_title ("red channel - 6 levels of representation")
plt.show()

figure = plt.figure(figsize=(10,10))
ax = figure.add_subplot (1,1,1)
ax.imshow(dataout_green_6)
ax.set_title ("green channel - 6 levels of representation")
plt.show()

figure = plt.figure(figsize=(10,10))
ax = figure.add_subplot (1,1,1)
ax.imshow(dataout_blue_6)
ax.set_title ("blue channel - 6 levels of representation")
plt.show()


figure = plt.figure(figsize=(10,10))
ax = figure.add_subplot (1,1,1)
ax.imshow(dataout_red_15)
ax.set_title ("red channel - 15 levels of representation")
plt.show()

figure = plt.figure(figsize=(10,10))
ax = figure.add_subplot (1,1,1)
ax.imshow(dataout_green_15)
ax.set_title ("green channel - 15 levels of representation")
plt.show()


figure = plt.figure(figsize=(10,10))
ax = figure.add_subplot (1,1,1)
ax.imshow(dataout_blue_15)
ax.set_title ("blue channel - 15 levels of representation")
plt.show()

six_level_all_channels = np.dstack((dataout_red_6, dataout_green_6, dataout_blue_6)).astype(np.uint8)
six_level_all_channels_rgb = cv2.cvtColor(six_level_all_channels, cv2.COLOR_BGR2RGB)

fifteen_level_all_channels = np.dstack((dataout_red_15, dataout_green_15, dataout_blue_15)).astype(np.uint8)
fifteen_level_all_channels_rgb = cv2.cvtColor(fifteen_level_all_channels, cv2.COLOR_BGR2RGB)

figure = plt.figure(figsize=(10,10))
ax = figure.add_subplot (1,1,1)
ax.imshow(six_level_all_channels_rgb)
ax.set_title ("all channels - 6 levels of representation")
plt.show()


figure = plt.figure(figsize=(10,10))
ax = figure.add_subplot (1,1,1)
ax.imshow(fifteen_level_all_channels_rgb)
ax.set_title ("all channels - 15 levels of representation")
plt.show()



figure = plt.figure(figsize=(10,10))
ax = figure.add_subplot (1,1,1)
ax.plot(distortion_red_6, 'r', label='Red channel distortion')
ax.plot(distortion_green_6, 'g', label='Green channel distortion')
ax.plot(distortion_blue_6, 'b', label='Blue channel distortion')
ax.legend()
ax.set_title ("6 levels of representation - Distortion over iterations")
plt.show()


figure = plt.figure(figsize=(10,10))
ax = figure.add_subplot (1,1,1)
ax.plot(distortion_red_15, 'r', label='Red channel distortion')
ax.plot(distortion_green_15, 'g', label='Green channel distortion')
ax.plot(distortion_blue_15, 'b', label='Blue channel distortion')
ax.legend()
ax.set_title ("15 levels of representation - Distortion over iterations")
plt.show()





#Q2

#2.a



folder_path = glob.glob("../given_data/LFW/*pgm")
faces = [cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2GRAY) for image in folder_path]


figure = plt.figure(figsize=(10,10))
ax = figure.add_subplot (1,1,1)
ax.imshow(faces[73], cmap ='gray')
ax.set_title ("Image number 73")
plt.show()

figure = plt.figure(figsize=(10,10))
ax = figure.add_subplot (1,1,1)
ax.imshow(faces[925], cmap ='gray')
ax.set_title ("Image number 925")
plt.show()

figure = plt.figure(figsize=(10,10))
ax = figure.add_subplot (1,1,1)
ax.imshow(faces[12819], cmap ='gray')
ax.set_title ("Image number 12819")
plt.show()

figure = plt.figure(figsize=(10,10))
ax = figure.add_subplot (1,1,1)
ax.imshow(faces[9587], cmap ='gray')
ax.set_title ("Image number 9587")
plt.show()



X = [np.reshape (face, [faces[0].size], 'F') for face in faces]
X = np.stack(X, axis=1)
mean = np.mean(X,axis=1)
mean_image = np.reshape(mean,faces[0].shape, 'F')

figure = plt.figure(figsize=(10,10))
ax = figure.add_subplot (1,1,1)
ax.imshow(mean_image, cmap ='gray')
ax.set_title ("Mean Image")
plt.show()

mu = np.reshape(mean_image,(4096,1), 'F')
Y = X-mu
covY = np.cov(Y)
print(covY.shape)

#2.b

eigenvalues, eigenvectors = np.linalg.eigh(covY)
k=10
indices = np.argsort(eigenvalues)[-k:]
eig_vals = eigenvalues[indices]
eig_vecs = eigenvectors[:,indices]

figure = plt.figure(figsize=(10,10))
ax = figure.add_subplot(1, 1, 1)
ax.plot(list(range(k)), eig_vals)
plt.title('k largest eigenvalues')
plt.grid()
plt.show()

eigenvector1 = np.reshape(eig_vecs[:,-1], (64,64) , 'F')
eigenvector2 = np.reshape(eig_vecs[:,-2], (64,64) , 'F')
eigenvector3 = np.reshape(eig_vecs[:,-3], (64,64) , 'F')
eigenvector4 = np.reshape(eig_vecs[:,-4], (64,64) , 'F')


figure = plt.figure(figsize=(10,10))
ax = figure.add_subplot (1,1,1)
ax.imshow(eigenvector1, cmap ='gray')
ax.set_title ("eigenvector1")
plt.show()

figure = plt.figure(figsize=(10,10))
ax = figure.add_subplot (1,1,1)
ax.imshow(eigenvector2, cmap ='gray')
ax.set_title ("eigenvector2")
plt.show()

figure = plt.figure(figsize=(10,10))
ax = figure.add_subplot (1,1,1)
ax.imshow(eigenvector3, cmap ='gray')
ax.set_title ("eigenvector3")
plt.show()

figure = plt.figure(figsize=(10,10))
ax = figure.add_subplot (1,1,1)
ax.imshow(eigenvector4, cmap ='gray')
ax.set_title ("eigenvector4")
plt.show()

#2.c

P = np.dot(np.transpose (eig_vecs), Y)
print (P.shape)


#2.d aux
def calc_x(image_num,eigen_vectors,P,mu):
    return np.reshape(np.matmul(eigen_vectors, P[:, image_num]) + np.squeeze(mu), (64, 64), 'F')

def calc_mse(faces,image,x):
    return int(np.sum((faces[image]-x)**2)/faces[image].size)
#2.d

x1 = calc_x(73,eig_vecs,P,mu)
x2 = calc_x(925,eig_vecs,P,mu)
x3 = calc_x(12819,eig_vecs,P,mu)
x4 = calc_x(9587,eig_vecs,P,mu)

mse1 = calc_mse(faces,73,x1)
mse2 = calc_mse(faces,925,x2)
mse3 = calc_mse(faces,12819,x3)
mse4 = calc_mse(faces,9587,x4)


figure = plt.figure(figsize=(10,10))
ax = figure.add_subplot (1,1,1)
ax.imshow(x1, cmap ='gray')
ax.set_title (f'Image 73 Restored, MSE is: {mse1}')
plt.show()

figure = plt.figure(figsize=(10,10))
ax = figure.add_subplot (1,1,1)
ax.imshow(x2, cmap ='gray')
ax.set_title (f'Image 925 Restored, MSE is: {mse2}')
plt.show()

figure = plt.figure(figsize=(10,10))
ax = figure.add_subplot (1,1,1)
ax.imshow(x3, cmap ='gray')
ax.set_title (f'Image 12819 Restored, MSE is: {mse3}')
plt.show()

figure = plt.figure(figsize=(10,10))
ax = figure.add_subplot (1,1,1)
ax.imshow(x4, cmap ='gray')
ax.set_title (f'Image 9587 Restored, MSE is: {mse4}')
plt.show()

#2.e

eigenvalues, eigenvectors = np.linalg.eigh(covY)
k_new=570
indices_new = np.argsort(eigenvalues)[-k_new:]
eig_vals_new = eigenvalues[indices_new]
eig_vecs_new = eigenvectors[:,indices_new]

P_new = np.dot(np.transpose (eig_vecs_new), Y)

x1_new = calc_x(73,eig_vecs_new,P_new,mu)
x2_new = calc_x(925,eig_vecs_new,P_new,mu)
x3_new = calc_x(12819,eig_vecs_new,P_new,mu)
x4_new = calc_x(9587,eig_vecs_new,P_new,mu)

mse1_new = calc_mse(faces,73,x1_new)
mse2_new = calc_mse(faces,925,x2_new)
mse3_new = calc_mse(faces,12819,x3_new)
mse4_new = calc_mse(faces,9587,x4_new)


figure = plt.figure(figsize=(10,10))
ax = figure.add_subplot (1,1,1)
ax.imshow(x1_new, cmap ='gray')
ax.set_title (f'Image 73 Restored, MSE is: {mse1_new}')
plt.show()

figure = plt.figure(figsize=(10,10))
ax = figure.add_subplot (1,1,1)
ax.imshow(x2_new, cmap ='gray')
ax.set_title (f'Image 925 Restored, MSE is: {mse2_new}')
plt.show()

figure = plt.figure(figsize=(10,10))
ax = figure.add_subplot (1,1,1)
ax.imshow(x3_new, cmap ='gray')
ax.set_title (f'Image 12819 Restored, MSE is: {mse3_new}')
plt.show()

figure = plt.figure(figsize=(10,10))
ax = figure.add_subplot (1,1,1)
ax.imshow(x4_new, cmap ='gray')
ax.set_title (f'Image 9587 Restored, MSE is: {mse4_new}')
plt.show()
