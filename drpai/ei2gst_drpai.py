import gc
import os
import argparse
import tflite_runtime.interpreter as tflite
import numpy as np
from loguru import logger as logging


def csv_2_bytearray(s: str) -> bytearray:
    """
    Converts a comma-separated string of hex values to a bytearray.

    Args:
        s (str): A string containing comma-separated hex values to be converted to a bytearray.

    Returns:
        bytearray: A bytearray representing the hex values, suitable for writing to a binary file.

    Examples:
        >>> csv_2_bytearray('0x84, 0xa0, 0x1c, 0xa6, 0x9e, 0x2c, 0xb1, 0xa3, 0x22, 0xa7, 0xa6, 0x23,')
        bytearray(b'\x84\xa0\x1c\xa6\x9e,\xb1\xa3"\xa7\xa6#')
    """
    c = 0
    array = []
    while c < len(s):
        i = s.find(",", c)
        substr = s[c:i] if i != -1 else s[c:]
        value = int(substr.strip(), 16)
        array.append(value)
        c += len(substr)+1
    r = bytearray(array)
    return r


def get_grid_anchors(interpreter: tflite.Interpreter, grids: list[int]) -> tuple[int, list]:
    """
    Retrieves grid anchors from the TensorFlow Lite interpreter.

    This method iterates through the tensors in the interpreter to find a tensor
    that matches the shape corresponding to one of the provided grids. It returns
    the grid size and the anchor values if a matching tensor is found.

    Args:
        interpreter (tflite.Interpreter): The TensorFlow Lite interpreter.
        grids (list[int]): A list of grid sizes to search for.

    Returns:
        (int, list[list[int]]): The first found grid size and its list of anchor values.

    Raises:
        ValueError: If no anchors are found for the provided grids in the interpreter.

    Example:
        >>> interpreter = tflite.Interpreter(model_path='yolov5.part2')
        >>> grids = [20, 40, 80]
        >>> __get_grid_anchors(interpreter, grids)
        [[116.0, 90.0], [156.0, 198.0], [373.0, 326.0]]
    """
    tensor_index = 0

    # Loop through all tensors
    # Note: ValueError is raised if tensor_index is out of bounds
    while True:
        try:
            # Get the tensor and its shape at the current index
            tensor = interpreter.get_tensor(tensor_index)
            shape = tensor.shape
            tensor_index += 1

            # Check if the tensor shape matches any of the provided grid sizes
            for g in grids:
                if np.array_equal(shape, [1, 3, g, g, 2]):
                    # Return the found grid and anchor values
                    return g, [list(tensor[0][j][0][0]) for j in range(3)]

        except ValueError:
            raise Exception("No anchors found for the provided grids not found in the interpreter.")


class EdgeImpulse2GstDRPAI:
    """
    Converts an EdgeImpulse deployment to GStreamer DRPAI plugin using a `model_name` and the `run()` function.

    Parameters:
        model_name (str): The name of the model to save.
    """

    def __init__(self, model_name: str, working_directory: str = '.'):
        """
        Initialises the class and recreates a directory with the name `model_name`

        Args:
            model_name (str): The name of the model to save.
            working_directory (str): The directory path that contains 'tflite-model' and 'model-parameters' directories.
        """
        self.var_list = dict()  # Dictionary to hold keys and values read from header files
        self.model_name = model_name
        self.working_directory = working_directory
        self.model_path = f"{working_directory}/{model_name}"
        self.model_classification = None    # This variable is filled after running gen_postprocess_params_txt()
        logging.info(f"Creating folder: {self.model_path}")
        os.makedirs(self.model_path, exist_ok=True)

    def __arrayname_2_filename(self, array_name: str) -> str:
        """
        Converts a C array declaration to a corresponding file name.

        Args:
            array_name (str): The array declaration name to be converted.

        Returns:
            str: The generated file name.

        Example:
            >>> self.model_name = 'model'
            >>> self.__arrayname_2_filename('unsigned char ei_ei_addrmap_intm_txt[] = {')
            'model/model_addrmap_intm.txt'
        """
        array_name = array_name.removeprefix("unsigned char ")
        array_name = array_name.removeprefix("ei_")
        array_name = array_name.replace("ei_", f"{self.model_name}_")
        array_name = array_name.replace("[]", "")
        array_name = array_name.replace("=", "")
        array_name = array_name.replace("{", "")
        partial_name = array_name.strip().split("_")
        return f"{self.model_path}/{'_'.join(partial_name[:-1])}.{partial_name[-1]}"

    def gen_drpai_model_files(self):
        """
        Extracts DRPAI model files by reading C arrays in the header file `tflite-model/drpai_model.h`
        and writing them down into binary files.

        It identifies array declarations and converts their data into bytearrays,
        which are then written to separate files. It also extracts and stores some metadata key-value pairs.

        Raises:
            AssertionError: If the file size does not match the array length.

        Example:
            >>> self.gen_drpai_model_files()
            Reading file: tflite-model/drpai_model.h
              Writing file: model/model_addrmap_intm.txt
              Writing file: model/drp_desc.bin
              Writing file: model/model_drpcfg.mem
              Writing file: model/drp_param.bin
              Writing file: model/aimac_desc.bin
              Writing file: model/model_weight.dat
              Writing file: model/model.part2
        """

        file_path = f"{self.working_directory}/tflite-model/drpai_model.h"
        logging.info("Reading file: " + file_path)
        output_file_size = 0        # the size of the last output file
        output_file_path = None     # the path of the last output file
        with open(file_path, "rt") as f:
            # Read the C header file line by line.
            for line in f:
                # Are we currently in the middle of writing a binary file?
                if output_file_path is None:
                    if "=" in line:
                        # This line is a declaration/assignment.
                        if "[]" in line:
                            # We have a new C array declaration.
                            # Generate the `output_file_path` from its name.
                            output_file_path = self.__arrayname_2_filename(line)
                            self.var_list[output_file_path] = bytearray()
                            logging.info("  Writing file: " + output_file_path)
                        else:
                            # We have a scalar variable declaration.
                            # Store it in the `var_list` dictionary.
                            line_sections = line.split(" ")
                            key = line_sections[-3]
                            value = line_sections[-1].removesuffix("\n").removesuffix(";")
                            self.var_list[key] = value
                            if key.endswith("_len"):
                                # Ensure the length of the last array is correct.
                                assert output_file_size == int(value), f"{output_file_path} seems to be corrupt."
                else:
                    # We are currently writing a binary file
                    if "}" in line:
                        # The C array declaration has ended, write it in a file and store its length.
                        with open(output_file_path, "wb") as writer:
                            writer.write(self.var_list[output_file_path])
                        output_file_size = len(self.var_list[output_file_path])
                        del self.var_list[output_file_path]
                        output_file_path = None
                        gc.collect()
                    else:
                        # The C array has not ended yet, so convert hex values to binary and append.
                        self.var_list[output_file_path] += csv_2_bytearray(line.removesuffix("\n"))

    def read_variables(self):
        """
        Reads and stores variables the `model_metadata.h` and `model_variables.h` header files.
        It also handles various formats and structures within the header files.

        This function must be called after `gen_drpai_model_files()`.
        """

        file_path = f"{self.working_directory}/model-parameters/model_metadata.h"
        logging.info("Reading file: " + file_path)
        struct_variables = None     # A list of variable names when they are grouped in a structure
        with open(file_path, "rt") as f:
            # Read the C header file line by line.
            for line in f:
                # By splitting the line into words, we can look for C language keywords
                line_sections = line.split()
                if len(line_sections) == 0:
                    continue    # Empty line, skip

                # Are we in the middle of a struct definition?
                if struct_variables is None:
                    if line_sections[0] == "#define":
                        if len(line_sections) == 3:
                            # It is a define macro, so it should follow the grammar `#define KEY VALUE`
                            # Let's add it to the dictionary of variables
                            self.var_list[line_sections[1]] = line_sections[2]
                    elif line_sections == ["typedef", "struct", "{"]:
                        # It is starting a struct definition, let's group all those variables in a list.
                        struct_variables = list()

                else:
                    # It is a member of a struct definition.
                    if len(line_sections) >= 2:
                        # The line probably looks like `type name;`, `type * name;`, `type *name;` or `} name;`
                        # We only care about the last word which is a name.
                        name = line_sections[-1].removeprefix("*").removesuffix(";")
                        if line_sections[0] == "}":
                            # The struct definition is finished
                            # Let's add all the variable names to the dictionary with a prefix of the struct name.
                            # Note: there are no values in the definition; we will write their values later.
                            for var_member in struct_variables:
                                self.var_list[f"{name}_{var_member}"] = None
                            struct_variables = None
                        else:
                            # The struct definition is not finished yet; let's just hold the variable name
                            struct_variables.append(name)

        file_path = f"{self.working_directory}/model-parameters/model_variables.h"
        logging.info("Reading file: " + file_path)
        struct_variables = list()   # A list of variable names when they are grouped in a structure
        with (open(file_path, "rt") as f):
            # Read the C header file line by line.
            for line in f:
                # By splitting the line into words, we can look for C language keywords
                line_sections = line.split()
                if len(line_sections) == 0:
                    continue    # Empty line, skip

                # Are we in the middle of a struct initialization?
                if len(struct_variables) == 0:
                    # We haven't started a struct initialization.
                    if "=" in line:
                        # We have a new assignment
                        if "[]" in line:
                            # It is defining an array
                            # Let's extract its contents into a list and save it in the `var_list`
                            name = line_sections[line_sections.index("=")-1].removesuffix("[]")
                            value = line[line.find("{")+1: line.find("}")]
                            value = value.replace("\"", "").replace(" ", "")
                            self.var_list[name] = value.split(",")
                        elif "ei_object_detection_nms_config_t" in line:
                            # This structure is defined in `edge-impulse-sdk` but we don't need to read it all
                            # I just steal the definition of and put it in the struct_variables.
                            struct_variables = [
                                "ei_object_detection_nms_config_t_confidence_threshold",
                                "ei_object_detection_nms_config_t_iou_threshold"
                            ]
                        elif "{" in line:
                            # It is starting a struct initialization. The line should look like `const TYPE NAME = {`
                            # We need the name of all variables inside the structure to assign values to them.
                            # So let's grab all the variable names that have the prefix of `TYPE_`
                            if line_sections[0] == "const":
                                # Delete the const keyword
                                del line_sections[0]
                            struct_variables = [k for k in self.var_list.keys() if k.startswith(line_sections[0])]
                        else:
                            # It is defining a variable. The line should look like `const TYPE NAME = VALUE;`
                            # Let's write it into the `var_list`
                            if line_sections[0] == "const":
                                # Delete the const keyword
                                del line_sections[0]
                            self.var_list[line_sections[1]] = line_sections[-1].removesuffix(";")
                else:
                    # We are in the middle of a struct initialization.
                    # The order of values match the order of variable names at the declaration time.
                    # Variable names are already selected in `struct_variables`.
                    # Let's assign values to the first item and remove. It will eventually become empty.
                    value = line_sections[0].removesuffix(",").removeprefix("\"").removesuffix("\"")
                    self.var_list[struct_variables[0]] = value
                    del struct_variables[0]

        # We also need to store the address and sizes of each model file in `var_list` dictionary.
        # They are included in `model/model_addrmap_intm.txt` in a format of `NAME HEX_ADDRESS HEX_SIZE` lines.
        file_path = f"{self.model_path}/{self.model_name}_addrmap_intm.txt"
        logging.info("Reading file: " + file_path)
        with open(file_path, "rt") as f:
            # Read the text file line by line.
            for line in f:
                line_sections = line.split()
                if len(line_sections) == 3:
                    # Store the address and size separately.
                    self.var_list[f"{line_sections[0]}_address"] = line_sections[1]
                    self.var_list[f"{line_sections[0]}_size"] = line_sections[2]

    def gen_data_in_list_txt(self):
        """
        Generates a `model/model_data_in_list.txt` file with model input details stored
        in the `var_list` dictionary. It ensures that all necessary variables are present before
        writing the file.

        This function must be called after `read_variables()`.

        Raises:
            AssertionError: If any required input items (`channels`, `width`, or `height`) are missing.
        """
        file_path = f"{self.model_path}/{self.model_name}_data_in_list.txt"

        # Retrieve required variables from var_list dictionary
        channels = self.var_list['ei_dsp_config_image_t_channels'].lower()
        address = self.var_list['data_in_address']
        width = self.var_list['EI_CLASSIFIER_INPUT_WIDTH']
        height = self.var_list['EI_CLASSIFIER_INPUT_HEIGHT']
        # Ensure all necessary variables are present
        assert channels is not None and width is not None and height is not None, \
            "The loaded model_variables.h doesn't have required input items."

        logging.info("  Writing file: " + file_path)
        with open(file_path, "wt") as f:
            f.write(f"Input_node_name: {channels}_data\n"
                    f"         Address: 0x{address}\n"
                    f"         Channel: {len(channels)}\n"
                    f"         Width  : {width}\n"
                    f"         Height : {height}")

    def gen_labels_txt(self):
        """
        Generates a `model_labels.txt` file with model labels stored in the `var_list` dictionary.
        It ensures that the number of labels matches the expected label count.

        This function must be called after `read_variables()`.

        Raises:
            AssertionError: If the number of labels does not match the expected label count.
        """
        file_path = f"{self.model_path}/{self.model_name}_labels.txt"

        # Retrieve labels from var_list dictionary
        labels = self.var_list['ei_classifier_inferencing_categories']
        # Ensure the number of labels matches the expected label count
        assert len(labels) == int(self.var_list['EI_CLASSIFIER_LABEL_COUNT']), "The loaded labels seems to be corrupt."

        logging.info("  Writing file: " + file_path)
        with open(file_path, "wt") as f:
            f.write("\n".join(labels))  # Write the labels, each on a new line

    def gen_data_out_list_txt(self):
        """
        Generates a `model_data_out_list.txt` file with model output details stored in the `var_list` dictionary.
        It ensures that there are output grids available.

        This function must be called after `read_variables()`.

        Raises:
            AssertionError: If no output grids are found in the `var_list`.
        """
        file_path = f"{self.model_path}/{self.model_name}_data_out_list.txt"

        # Retrieve output grid sizes from var_list dictionary
        grid_sizes = [k for k in self.var_list.keys() if k.startswith("NUM_GRID_")]
        # Ensure there are output grids available
        assert len(grid_sizes) > 0, "The loaded drpai_model.h doesn't have the required output grids."

        logging.info("  Writing file: " + file_path)
        with open(file_path, "wt") as f:
            for grid_size_param in grid_sizes:
                grid_size = self.var_list[grid_size_param]
                f.write(f"Output_node_name: {grid_size_param}\n"
                        f"         Address: 0\n"
                        f"         Channel: 0\n"
                        f"         Width  : {grid_size}\n"
                        f"         Height : {grid_size}\n")

    def gen_postprocess_params_txt(self):
        """
        Generates a `model_post_process_params.txt` file with post-processing parameters.

        This method creates a text file that includes the dynamic library file name for post-processing,
        the model version, and the IoU threshold based on the values stored in the `var_list`
        dictionary. It ensures that the necessary variables are present and supported.

        This function must be called after `read_variables()`.

        Raises:
            AssertionError: If the model classification or version is not supported, or if the IOU threshold is invalid.
        """
        file_path = f"{self.model_path}/{self.model_name}_post_process_params.txt"

        # Retrieve the model classification from var_list dictionary
        self.model_classification = self.var_list["EI_CLASSIFIER_OBJECT_DETECTION_LAST_LAYER"]
        post_process_library = None
        model_version = None
        iou_threshold = None

        # Determine the post-processing library and model version based on classification
        if "yolov5" in self.model_classification.lower():
            post_process_library = "libgstdrpai-yolo.so"
            model_version = "5"

            # Retrieve and validate the IoU threshold from var_list dictionary
            iou_threshold = self.var_list["ei_object_detection_nms_config_t_iou_threshold"].removesuffix("f")
            assert float(iou_threshold) >= 0, \
                "The loaded model_variables.h doesn't have the required output iou_threshold."

        elif "fomo" in self.model_classification.lower():
            # TODO: Add other libraries
            pass

        # Ensure the post-processing library and model version are supported
        assert post_process_library is not None, f"The script doesn't support classification '{self.model_classification}'."
        assert model_version is not None, f"The script doesn't support classification version '{self.model_classification}'."


        logging.info("  Writing file: " + file_path)
        with open(file_path, "wt") as f:
            f.write(f"[dynamic_library]\n{post_process_library}\n\n"
                    f"[yolo_version]\n{model_version}\n\n"
                    f"[iou_threshold]\n{iou_threshold}")

    def gen_anchors_txt(self):
        """
        Generates a `model_anchors.txt` file with anchor values.

        This method reads the YOLOv5 model, extracts anchor values for different grid sizes,
        and writes them to a text file.
        """
        model_path = f"{self.model_name}/yolov5.part2"

        logging.info("Reading file: " + model_path)

        # Initialize the TensorFlow Lite interpreter for the YOLOv5 model
        interpreter = tflite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()

        # Retrieve grid sizes from var_list dictionary
        grids = list()
        i = 0
        grid_var_key = f"NUM_GRID_{i+1}"
        while ei.var_list.__contains__(grid_var_key):
            grids.append(int(ei.var_list[grid_var_key]))
            i += 1
            grid_var_key = f"NUM_GRID_{i+1}"

        file_path = f"{self.model_path}/{self.model_name}_anchors.txt"
        logging.info("  Writing file: " + file_path)

        # Find anchor values and write them to the text file
        with open(file_path, "wt") as f:
            while len(grids) > 0:
                g, anchors = get_grid_anchors(interpreter, grids)
                if anchors is not None:
                    for a in anchors:
                        for b in a:
                            f.write(f"{b}\n")
                    grids.remove(g)

    def run(self):
        """
        Executes the full pipeline to generate necessary model files and parameters.

        This method runs a sequence of functions to:
        1. Generate DRPAI model binary and text files.
        2. Read and store variables from the model metadata and variable header files.
        3. Generate the `model_data_in_list.txt` file.
        4. Generate the `model_labels.txt` file.
        5. Generate the `model_data_out_list.txt` file.
        6. Generate the `model_post_process_params.txt` file.
        7. If the model classification is YOLO, generate the `model_anchors.txt` file.

        This method ensures that all necessary files and parameters are generated in the correct order.
        """
        self.gen_drpai_model_files()
        self.read_variables()           # Fills the `var_list` dictionary
        self.gen_data_in_list_txt()
        self.gen_labels_txt()
        self.gen_data_out_list_txt()
        self.gen_postprocess_params_txt()
        if "yolo" in self.model_classification.lower():
            self.gen_anchors_txt()


if __name__ == '__main__':
    """
    Main entry point for the EdgeImpulse2GstDRPAI script.
    
    This script initializes the `EdgeImpulse2GstDRPAI` class and runs the full pipeline to generate 
    necessary model files and parameters. It takes a single command-line argument `model_name` which 
    specifies the folder and prefix for the generated files.
    
    Command-line Arguments:
        model_name (str): The folder and prefix of files to create.
    
    Example:
        $ python3 ei2gst_drpai.py model
    """

    parser = argparse.ArgumentParser(prog='EdgeImpulse2GstDRPAI',
                                     description='EdgeImpulse DRPAI Deployment to GStreamer DRPAI plugin translator')
    parser.add_argument('model_name', help='The folder and prefix of files to create.')
    args = parser.parse_args()

    ei = EdgeImpulse2GstDRPAI(args.model_name)
    ei.run()
