instances = {}


def register_dataset_instances(name_instance, json_file_path: str, images_path: str):
    global instances
    instances[name_instance] = (json_file_path, images_path)


if __name__ == '__main__':
    register_dataset_instances('name', 'json', 'image')
    register_dataset_instances('name2', 'json2', 'image2')
    register_dataset_instances('name3', 'json3', 'image3')
    for name in instances.keys():
        print(f'name={name}')
    names = [name for name in instances.keys() if name in ('name3', 'name')]
    print(f'name={names}')
    print(instances)
