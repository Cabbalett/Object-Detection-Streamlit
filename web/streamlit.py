import streamlit as st
import io
import os

from mmdet.apis import init_detector, inference_detector

from PIL import Image
import numpy as np

st.set_page_config(layout="wide")


def main():
    st.title("My CV Project")

    st.subheader("Object Detection task with Faster RCNN model pretrained with COCO Dataset")
    cmd = 'wget https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
    # Specify the path to model config and checkpoint file
    if not os.path.isfile('faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'):
        with st.spinner("Downloading model parameters... It might take up to a minute"):
            os.system(cmd)
        
    config_file = '../mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
    checkpoint_file = 'faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
    model = init_detector(config_file, checkpoint_file, device='cpu')

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg","png"])

    if uploaded_file:
        image_bytes = uploaded_file.getvalue()
        image = Image.open(io.BytesIO(image_bytes))
        image = np.asarray(image)
        st.image(image, caption='Uploaded Image')

        if st.button("Click to Start Inference!"):
            with st.spinner("Inferencing..."):
                try:
                    result = inference_detector(model , image)
                except:
                    st.error("No object was detected!")
                    return
            result = model.show_result(image, result)

            st.image(result, caption='Result')
            st.success("Success!")


if __name__ == '__main__':
    main()