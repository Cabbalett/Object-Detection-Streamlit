# Object-Detection-Streamlit

## Setup

1. Make a new conda environment

        $ conda create -n mmdet python=3.7 -y
        $ conda activate mmdet

2. Install predefined libraries

        $ pip install -r requirements.txt

3. Git clone mmdetection on the home directory

        $ git clone https://github.com/open-mmlab/mmdetection.git
        $ cd mmdetection
        $ pip install -v -e .

4. Run the streamlit.py file

        $ cd ..
        $ streamlit run web/streamlit.py

4-1. If running on server

        $ cd ..
        $ streamlit run web/streamlit.py --browser.serverAddress localhost

