from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorflow.io import decode_image
import imageio
import cv2
from tqdm import tqdm

path = r".\runs\Mar25_11-08-44_DESKTOP-TNORDBI\events.out.tfevents.1648202926.DESKTOP-TNORDBI.7032.0"
ea = EventAccumulator(
    path,
    purge_orphaned_data=False,
    size_guidance={"images": 0},
)
ea.Reload()

imgs = ea.Images("images")

decoded_imgs = []
for i, img in enumerate(imgs,0):
    im = decode_image(img[2]).numpy()
    im = cv2.putText(img=im, text=str(i), org=(3,im.shape[0]), fontFace=3, fontScale=3, color=(255,0,0), thickness=5)
    
    decoded_imgs.append(im)
    
imageio.mimsave(path+'.gif', decoded_imgs, fps=3)