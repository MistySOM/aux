import os
import shutil


def csv_2_bytearray(s: str) -> bytearray:
    c = 0
    array = []
    while c < len(s):
        i = s.find(",", c)
        if i == -1:
            break
        value = int(s[c:i].strip(), 16)
        array.append(value)
        c = i + 1
    return bytearray(array)


class EdgeImpulse2drpai:

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
        output_file = None
        with open("tflite-model/drpai_model.h", "rt") as f:
            for line in f:
                if output_file is None:
                    if line.__contains__("[]"):
                        output_file_path = self.__arrayname_2_filename(line)
                        output_file = open(output_file_path, "wb")
                        print("  Writing file: " + output_file_path)
                else:
                    if line.__contains__("}"):
                        output_file.close()
                        output_file = None
                    else:
                        output_file.write(csv_2_bytearray(line))

        if output_file is not None:
            output_file.close()

    def read_variables(self):
        print("Reading file: model-parameters/model_metadata.h")
        struct_variables = None
        with open("model-parameters/model_metadata.h", "rt") as f:
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

        print("Reading file: model-parameters/model_variables.h")
        struct_variables = list()
        with open("model-parameters/model_variables.h", "rt") as f:
            for line in f:
                line_sections = line.split()
                if len(line_sections) == 0:
                    continue

                if len(struct_variables) == 0:
                    if line.__contains__("ei_classifier_inferencing_categories"):
                        line = line[line.find("{")+1: line.find("}")]
                        line = line.replace("\"", "").replace(" ", "")
                        self.var_list["ei_classifier_inferencing_categories"] = line.split(",")
                    else:
                        struct_variables = [k for k in self.var_list.keys() if k.startswith(line_sections[0])]
                else:
                    self.var_list[struct_variables[0]] = line_sections[0].removesuffix(",")
                    del struct_variables[0]


if __name__ == '__main__':
    ei = EdgeImpulse2drpai("yolov5")
    ei.gen_drpai_model_files()
    ei.read_variables()


