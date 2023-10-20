# mage-Analysis-Tkinter-App
A complex Tkinter application for analyzing images, detecting circles, and finding colonies in images. The app allows users to load images from a folder, display them, apply various processing operations, and visualize the results. But now it shows the original picture and not the edited one (called gfp_out oder image/output)
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
# from scipy import ndimage as ndi
from skimage.feature import peak_local_max


class App(Tk):
    def __init__(self):
        super().__init__()
        self.title("CustomTkinter complex_example.py")
        self.geometry(f"{1000}x600")

        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure((2, 3), weight=0)
        self.grid_rowconfigure((0, 1, 2), weight=1)
        
        self.z_frame = Frame(self)
        self.z_frame.grid(row=1, column=0)
        self.image_list = []
        self.image_number = 0

        self.button_back = Button(self.z_frame, text="<<", command=self.back)
        self.button_forward = Button(self.z_frame, text=">>", command=self.forward)
        self.button_back.grid(row=2, column=1, padx=(20, 10), pady=(10, 10))
        self.button_forward.grid(row=2, column=2, padx=(20, 10), pady=(10, 10))

        self.slider_progressbar_frame = Frame(self)
        self.slider_progressbar_frame.grid(row=1, column=1, padx=(20, 0), pady=(20, 0))
        self.slider_progressbar_frame.grid_columnconfigure(0, weight=1)
        self.slider_progressbar_frame.grid_rowconfigure(4, weight=1)
        self.slider_2 = Scale(self.slider_progressbar_frame, from_=0, to=4, orient=HORIZONTAL)
        self.slider_2.grid(row=0, column=1, padx=(20, 10), pady=(10, 10))
        self.dim_min_label = Label(self.slider_progressbar_frame, text="dim min").grid(row=0, column=0, pady=4, padx=4)
        #veränderung vom Bachelorcode von nöten
        self.slider_3 = Scale(self.slider_progressbar_frame, from_=0, to=4, orient=HORIZONTAL)
        self.slider_3.grid(row=1, column=1, padx=(20, 10), pady=(10, 10))
        self.dim_max_label = Label(self.slider_progressbar_frame, text="dim max").grid(row=1, column=0, pady=4, padx=4)
        #veränderung vom Bachelorcode von nöten
        self.slider_4 = Scale(self.slider_progressbar_frame, from_=0, to=4, orient=HORIZONTAL)
        self.slider_4.grid(row=2, column=1, padx=(20, 10), pady=(10, 10))
        self.sensitivity_label = Label(self.slider_progressbar_frame, text="sensitivity").grid(row=2, column=0, pady=4, padx=4)
        #veränderung vom Bachelorcode von nöten
        self.button_start = Button(self.slider_progressbar_frame, text="START COMPILING", command=self.button_start_command)
        self.button_start.grid (row=3, column=1, padx=(20, 10), pady=(10, 10))
        #veränderung vom Bachelorcode von nöten

        self.final_photo_frame = Frame(self)
        self.final_photo_frame.grid(row=1, column=3, padx=(20, 0), pady=(20, 0))
        self.final_photo = Button(self.final_photo_frame, text="final Photo", command=self.show_final_photo)
        self.final_photo.grid(row=2, column=1, padx=(20, 10), pady=(10, 10))
        self.my_label_output = Label(self.final_photo_frame)
        self.my_label_output.grid(row=1, column=1, columnspan=2, padx=(20, 10), pady=(10, 10))
        self.my_label = Label(self.z_frame)
        self.my_label.grid(row=1, column=1, columnspan=2, padx=(10, 10), pady=(10, 10))
        #veränderung vom Bachelorcode von nöten

        self.select_image_button = Button(self, text="Select Image", command=self.files)
        self.select_image_button.grid(row=0, column=1, padx=(10, 0), pady=(20, 20))

    def load_images_from_folder(self, folder_path):
            self.image_list = []
            for file in os.listdir(folder_path):
                if file.endswith('.tif'):
                    try:
                        image_path = os.path.join(folder_path, file)
                        image = Image.open(image_path)
                        image = image.resize((400, 375), Image.LANCZOS)
                        photo = ImageTk.PhotoImage(image)
                        self.image_list.append(photo)
                    except Exception as e:
                        print(f"Error: {e} - Could not open {file}")
            if self.image_list:
                self.my_label.config(image=self.image_list[self.image_number])
            #file wird hier abgespeichert und muss dann dem Code von Raphael übergeben werden        
        
    def forward(self):
        if self.image_number < len(self.image_list) - 1:
            self.image_number += 1
            self.my_label.config(image=self.image_list[self.image_number])

    def back(self):
        if self.image_number > 0:
            self.image_number -= 1
            self.my_label.config(image=self.image_list[self.image_number])

            
    #folder_name = files + "/*tif"
    #point_like = False #takes a long time if True

    def read_data(self, image_list):
        images = [cv2.imread (str(file), cv2.IMREAD_COLOR) for file in image_list]
        images_gray = [cv2.imread(str(file), cv2.IMREAD_GRAYSCALE) for file in image_list]
        return images, images_gray

    def find_circles(self, gray, output):

            # parameters for the circles function :
            # dp: inverse ratio of the image resolution (1 = input resolution )
            # minDist : minimum distance between the centres of the detected circles
            # param1 : some other parameter
            # param2 : the smaller this is, the more false circles may be detected
            # minRadius : minimum circle radius
            # maxRadius : maximum circle radius

            # detect circles in the image
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist =30, param1 =10, param2 =20, minRadius =10, maxRadius =20)
            # ensure at least some circles were found
        if circles is None:
            print("no circles found!")
        else:
                # convert the (x, y) coordinates and radius of the circles to integers
            circles = np.round(circles[0, :]).astype("int")
                # loop over the (x, y) coordinates and radius of the circles
            for(x , y , r ) in circles:
                    # draw the circle in the output image , then draw a rectangle
                    # corresponding to the center of the circle
                cv2.circle(output,(x , y), r, (255, 128, 0), 3)
                    #cv2. rectangle (output , (x - 5 , y - 5) , (x + 5 , y + 5) , (255 , 255 , 0), -1)
        return circles, output

    def find_radii(self, circles, xwidth):
        radii = circles[:,2]*1162.5/xwidth
        radii.sort() # sort from min to max
        size = radii*2
        mean_size = np.round(np.mean(size), 2)
        std_size = np.round(np.std(size), 2)
        return size, mean_size, std_size

    def find_colonies(self, point_like, i ,images_gray, gfp, circles, xwidth, ywidth, mean_size, std_size, find_intensity_max, find_intensity_avg):
        if point_like:
            im = images_gray[i]
            gfp, empty = find_intensity_max(im, gfp, circles, xwidth, ywidth)
            print(empty, "of", len(circles), "droplets are empty!")
            print("droplets are of size", mean_size , "\u03BCm with standard distribution of", std_size , "\u03BCm")
            print()

        else:
            intensities, gfp, colonies = find_intensity_avg(circles, xwidth, ywidth, gfp)
            print (colonies, "of", len(circles) , "droplets hold a colony")
            print ("the average intensity of the colonies is", np.round(np.mean(intensities),2))
            print ("droplets are of size", mean_size , "\u03BCm with standard distribution of", std_size, "\u03BCm")
            print()
        return

    def run_analysis(self, images, images_gray):
        for i in range(0, len(images)-1, 2):
            image = images [i+1]
            output = image.copy() #clone the image for the output
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert into gray scale image
            gray = cv2.medianBlur(gray, 5) #blur the image as this better works for the hough circles method
            gfp = images[i]
            gfp_out = gfp.copy()
            xwidth, ywidth = image.shape[0], image.shape[1]

            circles, output = self.find_circles(gray, output)
            size, mean_size, std_size = self.find_radii(circles, xwidth)

            self.find_colonies(False, i, images_gray, gfp, circles, xwidth, ywidth, mean_size, std_size)

            self.plot(gfp_out, gfp, image, output, size)

    def button_start_command(self):
        if self.image_list:
            images, images_gray = self.read_data(self.image_list)
            self.run_analysis(images, images_gray)

    def find_intensity_avg(self, circles, xwidth, ywidth, gfp):
        intensity = np.zeros((len(circles), 1))
        intensities = []
        thershold = 50
        colonies = 0
        for i, (x , y , r) in enumerate(circles):
            if x + r < xwidth and x - r > 0 and y + r < ywidth and y - r > 0:
                area = 0
                for xi in range(2* r):
                    for yi in range(2* r):
                        if(xi - r)**2+(yi - r)**2 <= r**2:
                            intensity [i ,0] += gfp[y + yi -r -1, x + xi -r -1, 0]
                            area += 1
                intensity[i ,0]/= area # rescaling by size$
                if intensity[i ,0] >= thershold:
                    intensities.append(intensity[i ,0])
                    colonies += 1
                    cv2.circle(gfp, (x , y), r, (0, 255, 0), 2)
                        #cv2.rectangle(gfp, (x - 5 , y - 5), (x + 5 , y + 5), (intensity[i ,0], intensity[i ,0], 0), -1)
        return intensities, gfp, colonies

    def find_intensity_max(self, im, gfp, circles, xwidth, ywidth):
        coordinates = peak_local_max(im, min_distance=10, threshold_rel=0.20)

        for c in range(len(coordinates)):
            cv2.circle(gfp, (coordinates[c ,1], coordinates[c,0]), 3, (0,255,0), 5)
        empty = 0
        for i, (x , y , r ) in enumerate(circles):
            occupied = False
            if x + r < xwidth and x - r > 0 and y + r < ywidth and y - r > 0:
                for xi in range(2*r):
                    for yi in range(2*r):
                        if(xi - r )**2+(yi - r )**2 <= r **2:
                            if[y + yi -r, x + xi - r] in coordinates.tolist():
                                occupied = True
            if occupied == False:
                empty += 1
                cv2.circle(gfp, (x,y),r ,(255,128,0), 4)
        return gfp, empty

    def plot(self, gfp, gfp_out, output, image, size):
        plt.imshow(gfp)
        plt.show()
        plt.imshow(gfp_out)
        plt.show()
        plt.imshow(output)
        plt.show()
        plt.imshow(image)
        plt.show()

    def files(self):
        folder_path = filedialog.askdirectory(initialdir="C:/", title="Select Folder")
        if folder_path:
            self.load_images_from_folder(folder_path)

    #def show_final_photo(self):
        #folder_path = self.image_list
        #if folder_path:
            #images, images_gray = self.read_data(folder_path)
            #for i in range(0, len(images) - 1, 2):
                #image = images[i + 1]
                #output = image.copy()  # Klone das Bild für die Ausgabe
                #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Konvertiere in ein Graustufenbild
                #gray = cv2.medianBlur(gray, 5)  # Verwische das Bild, da dies für die Hough-Circle-Methode besser funktioniert
                #gfp = images[i]
                #gfp_out = gfp.copy()
                #xwidth, ywidth = image.shape[0], image.shape[1]

                #circles, output = self.find_circles(gray, output)
                #size, mean_size, std_size = self.find_radii(circles, xwidth)

                #self.find_colonies(False, i, images_gray, gfp, circles, xwidth, ywidth, mean_size, std_size, self.find_intensity_max, self.find_intensity_avg)

                #self.plot(gfp_out, output, image, size)

            #if gfp_out is not None:
                #photo = Image.fromarray(gfp_out)
                #photo = ImageTk.PhotoImage(photo)
                #self.my_label_output.config(image=photo)
                #self.my_label_output.image = photo
            
    def show_final_photo(self):
        if self.image_list:
            selected_image = self.image_list[self.image_number]  # Get the currently selected image
            processed_image = self.process_image(selected_image)  # Process the selected image
            self.my_label_output.config(image=processed_image)  # Display the processed image
            

    # Method for processing the image
    def process_image(self, selected_image):
        img_pil = selected_image
        # Perform necessary image processing operations on 'img'
        processed_img = img_pil  # Placeholder for image processing operations
        #processed_photo = ImageTk.PhotoImage(processed_img)
        return processed_img
     
if __name__ == "__main__":
    app = App()
    app.mainloop()
