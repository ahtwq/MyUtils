# 使用pytorch训练深度模型时，会出现显存已满，但是GPU利用率较低的现象。这是因为DataLoader数据预处理可能存在问题，下面提供了一种解决方案。
# 使用了一个第三方库 prefetch_generator，继承DataLoader写了一个类DataLoaderX，把原来的DataLoader换成DataLoaderX。


from prefetch_generator import BackgroundGenerator
from torch.utils.data import DataLoader

class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())
        
loaders = {
    'train': DataLoaderX(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
    ),
    'valid': DataLoaderX(
        valid_set,
        batch_size=32,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
    ),
    'test': DataLoaderX(
        test_set,
        batch_size=32,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
    )
}
