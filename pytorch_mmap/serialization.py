import torch
import pickle
import os

_storages_types: list = [
    torch.DoubleStorage,
    torch.FloatStorage,
    torch.LongStorage,
    torch.IntStorage,
    torch.ShortStorage,
    torch.CharStorage,
    torch.ByteStorage,
    torch.BoolStorage,
]
_dtype_to_storage = {
    data_type(0).dtype: data_type for data_type in _storages_types
}

DEFAULT_PROTOCOL = 2


def save(obj, mmap_dir: str, pickle_module=pickle, pickle_protocol=DEFAULT_PROTOCOL) -> None:
    if not os.path.exists(mmap_dir):
        os.makedirs(mmap_dir)
    serialized_storages = {}

    def persistent_id(obj):
        if torch.is_storage(obj):
            assert obj.device.type == 'cpu', \
                f"can only dump storage on CPU to mmap. but here is a storage on '{obj.device.type}'"
            storage_type = getattr(torch, type(obj).__name__)
            obj_key = str(obj._cdata)
            serialized_storages[obj_key] = (obj, storage_type)

            return ('storage',
                    storage_type,
                    obj_key,
                    obj.size())
        return None

    # Dump model structure to pickle file
    with open(f"{mmap_dir}/model.pkl", 'wb') as model_file:
        pickler = pickle_module.Pickler(model_file, protocol=pickle_protocol)
        pickler.persistent_id = persistent_id
        pickler.dump(obj)

    # Write each tensor to a mmap file
    for key, value in serialized_storages.items():
        name = f'param_{key}'
        storage, storage_type = value
        dtype = storage_type(0).dtype
        mmap_storage = _dtype_to_storage[dtype].from_file(filename=f'{mmap_dir}/{name}', shared=True,
                                                          size=storage.size())
        mmap_storage.copy_(storage)
    return


def load(mmap_dir: str, pickle_module=pickle, **pickle_load_args):
    loaded_storages = {}

    # Load tensor from mmap files
    def load_tensor(data_type, key, size):
        name = f'{mmap_dir}/param_{key}'
        dtype = data_type(0).dtype

        storage = _dtype_to_storage[dtype].from_file(filename=name, shared=True, size=size)
        loaded_storages[key] = storage

    def persistent_load(saved_id):
        assert isinstance(saved_id, tuple)
        assert saved_id[0] == 'storage', \
            f"Unknown typename for persistent_load, expected 'storage' but got '{saved_id[0]}'"
        data = saved_id[1:]
        data_type, key, size = data
        if key not in loaded_storages:
            load_tensor(data_type, key, size)
        storage = loaded_storages[key]
        return storage

    with open(f"{mmap_dir}/model.pkl", 'rb') as model_file:
        unpickler = pickle_module.Unpickler(model_file, **pickle_load_args)
        unpickler.persistent_load = persistent_load
        result = unpickler.load()

    return result
