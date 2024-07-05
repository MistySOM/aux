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


def arrayname_2_filename(model_name: str, array_name: str) -> str:
    array_name = array_name.removeprefix("unsigned char ")
    array_name = array_name.removeprefix("ei_")
    array_name = array_name.replace("ei_", f"{model_name}_")
    array_name = array_name.replace("[]", "")
    array_name = array_name.replace("=", "")
    array_name = array_name.replace("{", "")
    partial_name = array_name.strip().split("_")
    return f"{model_name}/{'_'.join(partial_name[:-1])}.{partial_name[-1]}"


def gen_drpai_model_files(model_name: str):
    output_file = None
    with open("tflite-model/drpai_model.h", "rt") as f:
        for line in f:
            if output_file is None:
                if line.__contains__("[]"):
                    output_file_path = arrayname_2_filename(model_name, line)
                    output_file = open(output_file_path, "wb")
                    print("Writing file: " + output_file_path)
            else:
                if line.__contains__("}"):
                    output_file.close()
                    output_file = None
                else:
                    output_file.write(csv_2_bytearray(line))

    if output_file is not None:
        output_file.close()


def read_variables() -> dict[str, str]:
    output = dict()
    with open("model-parameters/model_metadata.h", "rt") as f:
        for line in f:
            if line.__contains__("#define"):
                line_sections = line.split()
                if len(line_sections) == 3:
                    output[line_sections[1]] = line_sections[2]
    return output


if __name__ == '__main__':
    model_name = "yolov5"
    shutil.rmtree(model_name, ignore_errors=True)
    os.mkdir(model_name)

    gen_drpai_model_files(model_name)
    vars = read_variables()
    print(vars)