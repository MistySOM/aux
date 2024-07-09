import gc
import os
import shutil


def csv_2_bytearray(s: str) -> bytearray:
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


class EdgeImpulse2GstDRPAI:

    def __init__(self, model_name: str):
        self.var_list = dict()
        self.model_name = model_name
        print(f"Creating folder: {model_name}")
        shutil.rmtree(model_name, ignore_errors=True)
        os.mkdir(model_name)

    def __arrayname_2_filename(self, array_name: str) -> str:
        array_name = array_name.removeprefix("unsigned char ")
        array_name = array_name.removeprefix("ei_")
        array_name = array_name.replace("ei_", f"{self.model_name}_")
        array_name = array_name.replace("[]", "")
        array_name = array_name.replace("=", "")
        array_name = array_name.replace("{", "")
        partial_name = array_name.strip().split("_")
        return f"{self.model_name}/{'_'.join(partial_name[:-1])}.{partial_name[-1]}"

    def gen_drpai_model_files(self):
        print("Reading file: model-parameters/model_metadata.h")
        last_file_length = 0
        output_file_path = None
        with open("tflite-model/drpai_model.h", "rt") as f:
            for line in f:
                if output_file_path is None:
                    if line.__contains__("="):
                        if line.__contains__("[]"):
                            output_file_path = self.__arrayname_2_filename(line)
                            self.var_list[output_file_path] = bytearray()
                            print("  Writing file: " + output_file_path)
                        else:
                            line_sections = line.split(" ")
                            key = line_sections[-3]
                            value = line_sections[-1].removesuffix("\n").removesuffix(";")
                            self.var_list[key] = value
                            if key.endswith("_len"):
                                assert last_file_length == int(value), f"{output_file_path} seems to be corrupt."
                else:
                    if line.__contains__("}"):
                        with open(output_file_path, "wb") as writer:
                            writer.write(self.var_list[output_file_path])
                        last_file_length = len(self.var_list[output_file_path])
                        del self.var_list[output_file_path]
                        output_file_path = None
                        gc.collect()
                    else:
                        self.var_list[output_file_path] += csv_2_bytearray(line.removesuffix("\n"))

    def read_variables(self):
        file_path = "model-parameters/model_metadata.h"
        print("Reading file: " + file_path)
        struct_variables = None
        with open(file_path, "rt") as f:
            for line in f:
                line_sections = line.split()
                if len(line_sections) == 0:
                    continue

                if struct_variables is None:
                    if line_sections[0] == "#define":
                        if len(line_sections) == 3:
                            self.var_list[line_sections[1]] = line_sections[2]
                    elif line_sections == ["typedef", "struct", "{"]:
                        struct_variables = list()

                else:
                    if len(line_sections) >= 2:
                        name = line_sections[-1].removeprefix("*").removesuffix(";")
                        if line_sections[0] == "}":
                            for var_member in struct_variables:
                                self.var_list[f"{name}_{var_member}"] = None
                            struct_variables = None
                        else:
                            struct_variables.append(name)

        file_path = "model-parameters/model_variables.h"
        print("Reading file: " + file_path)
        struct_variables = list()
        with (open(file_path, "rt") as f):
            for line in f:
                line_sections = line.split()
                if len(line_sections) == 0:
                    continue

                if len(struct_variables) == 0:
                    if line.__contains__("ei_classifier_inferencing_categories[]"):
                        line = line[line.find("{")+1: line.find("}")]
                        line = line.replace("\"", "").replace(" ", "")
                        self.var_list["ei_classifier_inferencing_categories"] = line.split(",")
                    else:
                        struct_variables = [k for k in self.var_list.keys() if k.startswith(line_sections[0])]
                else:
                    value = line_sections[0].removesuffix(",").removeprefix("\"").removesuffix("\"")
                    self.var_list[struct_variables[0]] = value
                    del struct_variables[0]

        file_path = f"{self.model_name}/{self.model_name}_addrmap_intm.txt"
        print("Reading file: " + file_path)
        struct_variables = list()
        with open(file_path, "rt") as f:
            for line in f:
                line_sections = line.split()
                if len(line_sections) == 3:
                    self.var_list[f"{line_sections[0]}_address"] = line_sections[1]
                    self.var_list[f"{line_sections[0]}_size"] = line_sections[2]

    def gen_data_in_list_txt(self):
        file_path = f"{self.model_name}/{self.model_name}_data_in_list.txt"
        channels = self.var_list['ei_dsp_config_image_t_channels'].lower()
        address = self.var_list['data_in_address']
        width = self.var_list['EI_CLASSIFIER_INPUT_WIDTH']
        height = self.var_list['EI_CLASSIFIER_INPUT_HEIGHT']
        assert channels is not None and width is not None and height is not None, \
            "The loaded model_variables.h doesn't have required input items."

        print("  Writing file: " + file_path)
        with open(file_path, "wt") as f:
            f.write(f"Input_node_name: {channels}_data\n"
                    f"         Address: 0x{address}\n"
                    f"         Channel: {len(channels)}\n"
                    f"         Width  : {width}\n"
                    f"         Height : {height}")

    def gen_labels_txt(self):
        file_path = f"{self.model_name}/{self.model_name}_labels.txt"
        labels = self.var_list['ei_classifier_inferencing_categories']
        assert len(labels) == int(self.var_list['EI_CLASSIFIER_LABEL_COUNT']), "The loaded labels seems to be corrupt."

        print("  Writing file: " + file_path)
        with open(file_path, "wt") as f:
            f.write("\n".join(labels))

    def gen_data_out_list_txt(self):
        file_path = f"{self.model_name}/{self.model_name}_data_out_list.txt"
        grid_sizes = [k for k in self.var_list.keys() if k.startswith("NUM_GRID_")]
        assert len(grid_sizes) > 0, "The loaded drpai_model.h doesn't have the required output grids."

        print("  Writing file: " + file_path)
        with open(file_path, "wt") as f:
            for grid_size_param in grid_sizes:
                grid_size = self.var_list[grid_size_param]
                f.write(f"Output_node_name: {grid_size_param}\n"
                        f"         Address: 0\n"
                        f"         Channel: 0\n"
                        f"         Width  : {grid_size}\n"
                        f"         Height : {grid_size}\n")

    def run(self):
        self.gen_drpai_model_files()
        self.read_variables()
        self.gen_data_in_list_txt()
        self.gen_labels_txt()
        self.gen_data_out_list_txt()


if __name__ == '__main__':
    ei = EdgeImpulse2GstDRPAI("yolov5")
    ei.run()
