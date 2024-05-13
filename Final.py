import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math
import numpy as np
block_size=8
PEAK=255
#convert to bmp file
pngImage=Image.open('image1.png')
bmpImage=pngImage.convert('RGB')
bmpImage.save('image1.bmp')
print('bmp image saved successfully')
# read image
image = cv2.imread('image1.bmp')
block_size=8#size =8
mValues=[1,2,3,4]
#----------------------------compress image component--------------------------------#
def compressComponent(component,m): 
    rows, cols = component.shape
    # Initialize the compressed component matrix
    compressedComponent = np.zeros((rows// block_size * m, cols// block_size * m))  
    l=0#row interate
    k=0# column
    for i in range(0,rows,block_size):
        k=0#the iteration differ from compressed and the original component
        for j in range(0,cols,block_size):
            #devide into 8x8 pixel blocks
            block = component[i:i + block_size, j:j + block_size]
            # Compute 2D DCT
            dctBlock = cv2.dct(block.astype(np.float32))
            # Quantize the DCT coefficients
            compressedComponent[l:l+m,k:k+m] = dctBlock[:m, :m]
            k+=m
        l+=m#nex row
    return compressedComponent
#--------------------compress for all m------------------#
def compress(originalImage,m):
    #split the three component of the original image
    blueComponent = originalImage[:,:,0]
    greenComponent = originalImage[:,:,1]
    redComponent = originalImage[:,:,2]
    
    #compress each component
    combressedRed=compressComponent(redComponent,m)
    combressedGreen=compressComponent(greenComponent,m)
    combressedBlue=compressComponent(blueComponent,m)
    return combressedRed,combressedGreen,combressedBlue
#----------------------------decompress image component--------------------------------#
def decompressComponent(compressedComponent,m):
     #initialize the compressed component matrix
    #print(compressedComponent)
    component = np.zeros((1080, 1920), dtype=compressedComponent.dtype)
    rows, cols = compressedComponent.shape
    k=0#iterate on rows
    l=0#column to add block size not m
    for i in range(0,rows,m):
        l=0
        for j in range(0,cols,m):
            #devide into 8x8 pixel blocks
            block=np.zeros((block_size,block_size),dtype=compressedComponent.dtype)
            block[0:m,0:m] = compressedComponent[i:i + m, j:j + m]
            idctBlock = cv2.idct(block.astype(np.float32))
            component[k:k + block_size, l:l + block_size] = idctBlock
            l=l+block_size# move column
        k=k+block_size#move row
    return component
#--------------------compress for all m------------------#
def decompress(compressedImage,m):
    #split the three component of the original image
    blueCompressedComponent = compressedImage[:,:,0]
    greenCompressedComponent = compressedImage[:,:,1]
    redCompressedComponent = compressedImage[:,:,2]
    #compress each component
    decombressedRed=decompressComponent(redCompressedComponent,m)
    decombressedGreen=decompressComponent(greenCompressedComponent,m)
    decombressedBlue=decompressComponent(blueCompressedComponent,m)
    decompressedImage=cv2.merge((decombressedBlue,decombressedGreen,decombressedRed))

    return decompressedImage
#----------------------- testing -------------------------#
rows, cols, channels = image.shape
PSNR=[]
# Calculate the size of the image
image_size_original = rows * cols * channels
for x in mValues:
    Rcompressed,Gcompressed,Bcompressed=compress(image,x)
    compressedImage=cv2.merge((Bcompressed,Gcompressed,Rcompressed))
    cv2.imwrite(f"compressedImage{x}.jpg",compressedImage)#save compressed version

    # Calculate the size of the image
    rows, cols, channels = compressedImage.shape
    image_size_compressed = rows * cols * channels
    print(f"at m = {x} image size is {image_size_compressed} original size is {image_size_original}")
    #####################################################################
    decompressedImage=decompress(compressedImage,x)#call compress image
    decompressedImage_uint8 = cv2.convertScaleAbs(decompressedImage)#scale it
    psnr = cv2.PSNR(image, decompressedImage_uint8)#calculate psnr
    PSNR.append(psnr)#add to the array
    cv2.imwrite(f"decompressedImage{x}.jpg",decompressedImage)
    
# print(PSNR)
# Plot the curve displaying PSNR against m

plt.plot(mValues, PSNR,color="red")
plt.xlabel('m')
plt.ylabel('PSNR')
plt.title('PSNR vs. m')
plt.grid(True)
plt.show()