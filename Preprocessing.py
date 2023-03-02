import glob
import cv2
import os
from skimage.feature import hog
i=1
#selecting the path of images
path=(r'C:\Users\user\Desktop\MCA\SEM 3\Project\Images\With Helmet\*.*')
try:
    # creating a folder named data
    if not os.path.exists('With Helmet'):
        os.makedirs('With Helmet')
    # if not created then raise error
except OSError:
    print('Error: Creating directory of data')

for file in glob.glob(path):
    img=cv2.imread(file)
    img2= cv2.resize(img, (400,400))
    
    # Path of output folder
    out_path=r"C:\Users\user\Desktop\noi\with helmet"
   
    #conversion of original images into RGB
  
    c=cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)
    
    #conversion of RGB images into Gray Scale
    gray = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
    
    # Otsu thresholding
    ret,img = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    # Sobel edge detection
   
    img_new = cv2.Sobel(src=img2, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)
    img2 = cv2.medianBlur(img, 5)
    
    fd, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), visualize=True)
    
    # writing the extracted images
    cv2.imwrite(os.path.join(out_path, 'crop_head'+str(i)+'.jpg'), hog_image)
    i=i+1
    k=cv2.waitKey(1000)
# show how many frames are created
print(i-1, "number of Heads")
if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows() 