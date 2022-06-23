import cv2
import os
import random


'''
def small_img_prepare(index):
    fp1 = "./raw_data/" + str(index) + ".png"
    fp2 = "./raw_data/" + str(index+10) + ".png"
    img1 = cv2.imread(fp1, 0)  # read a gray scale image
    img2 = cv2.imread(fp2, 0)

    if not os.path.exists("./splitted_data/"+str(index)):
        os.mkdir("./splitted_data/"+str(index))
        for i in range(6):
            for j in range(5):
                new_img = img1[i*200:(i+1)*200, j*200:(j+1)*200]
                cv2.imwrite("./splitted_data/"+str(index)+"/" +
                            str(i*5+j)+str(0)+".png", new_img)
                img_extention(new_img, index, i*5+j)

    if not os.path.exists("./splitted_data/"+str(index + 10)):
        os.mkdir("./splitted_data/"+str(index + 10))
        for i in range(6):
            for j in range(5):
                new_img = img2[i*200:(i+1)*200, j*200:(j+1)*200]
                cv2.imwrite("./splitted_data/"+str(index+10)+"/" +
                            str(i*5+j)+str(0)+".png", new_img)
                img_extention(new_img, index+10, i*5+j)

    for i in range(30):
        x = random.randrange(800)
        y = random.randrange(1000)
        if x % 200 == 0 or y % 200 == 0:
            i -= 1
            continue
        new_img1 = img1[y:y+200, x:x+200]
        new_img2 = img2[y:y+200, x:x+200]
        cv2.imwrite("./splitted_data/"+str(index)+"/" +
                    str(i+30)+str(0)+".png", new_img1)
        cv2.imwrite("./splitted_data/"+str(index+10)+"/" +
                    str(i+30)+str(0)+".png", new_img2)
        img_extention(new_img1, index, i+30)
        img_extention(new_img2, index+10, i+30)
'''


def img_prepare():
    s = set()
    i = 0
    while i < 90:
        x = random.randrange(1000-256)
        y = random.randrange(1200-256)
        l = len(s)
        s.add((x, y))
        if len(s) == l:
            continue
        i += 1

    for index in range(10):
        index += 1
        fp1 = "./raw_data/" + str(index) + ".png"
        fp2 = "./raw_data/" + str(index+10) + ".png"
        img1 = cv2.imread(fp1, 0)  # read a gray scale image
        img2 = cv2.imread(fp2, 0)

        if not os.path.exists("./splitted_data/"+str(index)):
            os.mkdir("./splitted_data/"+str(index))
        if not os.path.exists("./splitted_data/"+str(index + 10)):
            os.mkdir("./splitted_data/"+str(index + 10))

        i = 0
        for (x,y) in s:
            new_img1 = img1[y:y+256, x:x+256]
            new_img2 = img2[y:y+256, x:x+256]
            cv2.imwrite("./splitted_data/"+str(index)+"/" +
                        str(i)+str(0)+".png", new_img1)
            cv2.imwrite("./splitted_data/"+str(index + 10)+"/" +
                        str(i)+str(0)+".png", new_img2)
            img_extention(new_img1, index, i)
            img_extention(new_img2, index+10, i)
            i += 1


def img_extention(img, index, i):
    cv2.imwrite("./splitted_data/"+str(index)+"/" +
                str(i) + str(1)+".png", cv2.flip(img, -1))
    cv2.imwrite("./splitted_data/"+str(index)+"/" +
                str(i) + str(2)+".png", cv2.flip(img, 0))
    cv2.imwrite("./splitted_data/"+str(index)+"/" +
                str(i) + str(3)+".png", cv2.flip(img, 1))
    cv2.imwrite("./splitted_data/"+str(index)+"/" +
                str(i) + str(4)+".png", cv2.transpose(img))
    cv2.imwrite("./splitted_data/"+str(index)+"/"+str(i) +
                str(5)+".png", cv2.flip(cv2.transpose(img), 1))
    cv2.imwrite("./splitted_data/"+str(index)+"/"+str(i) +
                str(6)+".png", cv2.flip(cv2.transpose(img), 0))
    cv2.imwrite("./splitted_data/"+str(index)+"/"+str(i) +
                str(7)+".png", cv2.flip(cv2.transpose(img), -1))


if __name__ == "__main__":
    if not os.path.exists("./splitted_data"):
        os.mkdir("./splitted_data")

    img_prepare()
