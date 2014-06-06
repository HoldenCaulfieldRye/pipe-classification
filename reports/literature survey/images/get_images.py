import os, json

def all_labels(dirlist):
  images = {'perfect':[]}
  for filename in dirlist:
    if not filename.endswith('.dat'): continue
    with open(filename) as f:
      lines = f.readlines()
      lines = [line.strip() for line in lines]
      if lines == []:
        images['perfect'].append(filename)
      for line in lines:
        if line not in images.keys():
          images[line] = []
        images[line].append(filename)
  return images

def find_them():

  back = os.getcwd()
  img_dir = '*'
  while not os.path.exists(img_dir):
    img_dir = 'data2/ad6813/pipe-data/Redbox/raw_data/dump'  # raw_input('path to images? ')

  os.chdir(img_dir)
  dirlist = os.listdir(os.getcwd())

  images = all_labels(dirlist)

  os.chdir(back)
  json.dump(images, open('images.txt','w'))

  return images
    

if __name__ == '__main__':

  find_them()
