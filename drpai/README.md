# EdgeImpulse DRPAI Deployment to GStreamer DRPAI plugin translator

The python script is a translation layer to make the EdgeImpulse deployment usable for the 
[Gstreamer DRPAI Plugin](https://github.com/MistySOM/gstreamer1.0-drpai). 

> [!NOTE]
> There is no dependency and virtual environment required for the script to run.

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

2. Run the script with one parameter which is the name of the folder and prefix of files to create:
   
   ```bash
   python3 ei2gst-drpai.py yolov5
   ```

   > [!TIP]
   > If the script is run correctly, it would generate the folder with the following contents:
   >
   > ```
   > . 
   > ├── model-parameters
   > │   ├── model_metadata.h
   > │   └── model_variables.h
   > ├── tflite-model
   > │   └── drpai_model.h
   > ├── yolov5
   > │   ├── aimac_desc.bin
   > │   ├── drp_desc.bin
   > │   ├── drp_param.bin
   > │   ├── yolov5_addrmap_intm.txt
   > │   ├── yolov5_data_in_list.txt
   > │   ├── yolov5_data_out_list.txt
   > │   ├── yolov5_drpcfg.mem
   > │   ├── yolov5_labels.txt
   > │   ├── yolov5.part2
   > │   └── yolov5_weight.dat
   > └── ei2gst-drpai.py 
   > ```
   
3. As each model needs a different way of interpretation, 
   create a file in the model's directory with the name `<model>/<model>_post_process_params.txt`
   to adjust the post-processing parameters.

   > [!TIP]
   > For example, YOLOv5 post-processing parameters is a file `yolov5/yolov5_post_process_params.txt` and contains the following lines: 
   >
   > ```bash
   > [dynamic_library]
   > libgstdrpai-yolo.so
   > 
   > [best_class_prediction_algorithm]
   > sigmoid
   >   
   > [anchor_divide_size]
   > none
   > ```
   
   