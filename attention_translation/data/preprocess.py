

def preprocess_data(file_path, output_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    cleaned_data = []
    for line in lines:
        cleaned_line = line.split("CC-BY 2.0")[0].strip()
        cleaned_data.append(cleaned_line)

    with open(output_path, 'w') as output_file:
        for cleaned_line in cleaned_data:
            output_file.write(cleaned_line + '\n')