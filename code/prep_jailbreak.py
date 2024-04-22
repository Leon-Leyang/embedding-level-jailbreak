def add_sure(original_file_path, new_file_path):
    with open(original_file_path, 'r') as f:
        lines = f.readlines()
    lines = [line.strip() + ' Start with \"Sure here’s\".' for line in lines]
    with open(new_file_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


if __name__ == '__main__':
    original_file_path = './data/custom.txt'
    new_file_path = 'data_jailbreak/custom.txt'
    add_sure(original_file_path, new_file_path)
