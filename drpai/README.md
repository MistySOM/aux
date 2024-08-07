# 'EdgeImpulse DRPAI Deployment' to<br>'GStreamer DRPAI Plugin' translator

The python script is a translation layer to make the EdgeImpulse deployment usable for the 
[Gstreamer DRPAI Plugin](https://github.com/MistySOM/gstreamer1.0-drpai). 

To run the script, follow the instructions below:

1. Extract the deployment file in a way that the working directory would like below:
   
   ```
   . 
   ├── model-parameters
   │   ├── model_metadata.h
   │   └── model_variables.h
   ├── tflite-model
   │   └── drpai_model.h
   └── ei2gst-drpai.py 
   ```
    
   As an example, the `yolov5-ei-sample.tar.gz` is provided for extracting. 

2. Prepare a Python virtual environment and install the required dependencies in `requirements.txt`:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip3 install -r requirements.txt
   ```

3. Run the script with one parameter which is the name of the folder and prefix of files to create:
   
   ```bash
   python3 ei2gst_drpai.py yolov5
   ```

   <details>
      <summary>Example Output</summary>

      If the script is run correctly, it would generate the folder with the following contents:
      ```
      . 
      ├── model-parameters
      │   ├── model_metadata.h
      │   └── model_variables.h
      ├── tflite-model
      │   └── drpai_model.h
      ├── yolov5
      │   ├── aimac_desc.bin
      │   ├── drp_desc.bin
      │   ├── drp_param.bin
      │   ├── yolov5_addrmap_intm.txt
      │   ├── yolov5_anchors.txt
      │   ├── yolov5_data_in_list.txt
      │   ├── yolov5_data_out_list.txt
      │   ├── yolov5_drpcfg.mem
      │   ├── yolov5_labels.txt
      │   ├── yolov5.part2
      │   ├── yolov5_post_process_params.txt
      │   └── yolov5_weight.dat
      ├── ei2gst-drpai.py   
      └── requirements.txt
      ```
   </details>
   
