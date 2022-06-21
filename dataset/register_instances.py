from typing import List


class RegisterDataset:
    def __init__(self):
        self.instances = {}

    def register_dataset_instances(self, name_instance, json_file_path: str, images_path: str):
        self.instances[name_instance] = (json_file_path, images_path)

    def get_all_instances(self):
        return self.instances

    def get_instances(self, insts: List) -> List:
        ret_insts = [self.instances[name_insts] for name_insts in self.instances.keys() if name_insts in insts]
        return ret_insts


if __name__ == '__main__':
    rd = RegisterDataset()
    rd.register_dataset_instances('name', 'json', 'image')
    rd.register_dataset_instances('name2', 'json2', 'image2')
    rd.register_dataset_instances('name3', 'json3', 'image3')
    instances = rd.get_instances(['name2', 'name'])
    print(instances)
    # for name in instances.keys():
    #     print(f'name={name}')
    # names = [name for name in instances.keys() if name in ('name3', 'name')]
    # print(f'name={names}')
    # print(instances)
