import numpy as np
import streamlit as st
from PIL import Image
import cv2
def main():
    st.title("Image Processing App")

    uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg"])
    background_file =st.file_uploader("Choose one more file", type=["jpg", "jpeg"])
    if (uploaded_file is not None and background_file is not None):
        image = Image.open(uploaded_file)
        background_im=Image.open(background_file)
            # Display processed images
        st.image(image, caption="Original Image", use_column_width=True)
        st.image(background_im, caption="Background", use_column_width=True)
        # Convert PIL Image to OpenCV format
        my_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        background_im=Image.open(background_file)
        background_img = cv2.cvtColor(np.array(background_im), cv2.COLOR_RGB2BGR)
        # Your image processing code here
        width = int(my_img.shape[1])
        height = int(my_img.shape[0])
        dsize = (width, height)

        mask = np.zeros(my_img.shape[:2], np.uint8)

    
   
        mask_condition = ((my_img[:, :, 0] >= 0) & (my_img[:, :, 0] <=255) &
                        (my_img[:, :, 1] >= 10) & (my_img[:, :, 1] <= 250)&
                        (my_img[:, :, 2] >= 0) & (my_img[:, :, 2] <= 255))

        mask[mask_condition] = cv2.GC_PR_FGD

        background_model = np.zeros((1, 65), np.float64)
        foreground_model = np.zeros((1, 65), np.float64)
        

        cv2.grabCut(my_img, mask,None, background_model, foreground_model, 100, cv2.GC_INIT_WITH_MASK)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        no_bg_my = my_img * mask2[:, :, np.newaxis]
        
        
        print()
        background_img = cv2.resize(background_img, dsize)
        background_img[mask2 != 0] = [0, 0, 0]

        final_img = no_bg_my + background_img
        
        my_final_img_rgb = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)

        my_img_rgb=cv2.cvtColor(my_img, cv2.COLOR_BGR2RGB)
            # Display processed images
        st.image(my_img_rgb, caption="Original Image", use_column_width=True)
        st.image(background_im, caption="New Background", use_column_width=True)
        st.image(my_final_img_rgb, caption="Final Image", use_column_width=True)

if __name__ == "__main__":
    main()
