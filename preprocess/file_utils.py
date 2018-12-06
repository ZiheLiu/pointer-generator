def read_file_to_string(filename):
    with open(filename, 'r') as fin:
        return fin.read()


def write_string_to_file(filename, content):
    with open(filename, 'w') as fout:
        fout.write(content)
