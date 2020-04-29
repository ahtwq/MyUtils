from PIL import Image


def join(paths, flag='horizontal'):
    imgs = []
    for path in paths:
        img = Image.open(path)
        imgs += [img]
    if flag == 'horizontal':
        width = sum([img.size[0] for img in imgs])
        height = img.size[1]
        joint = Image.new('RGB', (width, height))
        x_aixs = 0
        for i,img in enumerate(imgs):
            loc = (x_aixs, 0)
            joint.paste(img, loc)
            x_aixs += img.size[0]
        joint.save('horizontal.png')



paths = ['train.jpg', 'valid.jpg', 'test.jpg']
join(paths)

