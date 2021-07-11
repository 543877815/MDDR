import lmdb
import cv2
import numpy as np

if __name__ == '__main__':
    filename = 'G:\\LSUN\\data\\bedroom_val_lmdb'
    env = lmdb.open(filename)
    txn = env.begin()
    for key, value in txn.cursor():  # 遍历
        print(key, value)
        img = cv2.imdecode(
            np.fromstring(value, dtype=np.uint8), 1)
        cv2.imshow("img", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        with open('./1.png', 'wb') as fp:
            fp.write(value)
    env.close()