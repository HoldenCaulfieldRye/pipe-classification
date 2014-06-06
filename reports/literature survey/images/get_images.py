import os, json

def just_a_few(dirlist, images):
  for filename in dirlist:
    if filename.endswith('.jpg'): continue
    with open(filename) as f:
      lines = f.readlines()
      lines = [line.strip() for line in lines]
      if 'SoilContaminationHighRisk' in lines:
        images['soilcontam'].append(filename)
      elif 'PoorPhoto' in lines:
        images['poor'].append(filename)
      elif 'WaterContaminationHighRisk' in lines:
        images['watercontam'].append(filename)
      elif 'NoClampUsed' in lines:
        images['noclamp'].append(filename)
      elif lines == []:
        images['perfect'].append(filename)

      if all(len(images[key]) for key in images.keys()):
        break
  return images

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

def find_them(return_dict):

  back = os.getcwd()
  img_dir = '*'
  while not os.path.exists(img_dir):
    img_dir = raw_input('path to images? ')

  os.chdir(img_dir)
  dirlist = os.listdir(os.getcwd())

  images = all_labels(dirlist)

  if return_dict: return images
  else:
    os.chdir(back)
    json.dump(images, open('images.txt','w'))
    

if __name__ == '__main__':

  return_dict = raw_input("return dict or save it to txt? [R/S] ")
  if return_dict == 'R': return_dict = True
  else: return_dict = False

  find_them(return_dict)
