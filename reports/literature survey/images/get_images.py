import os, json

if __name__ == '__main__':

  dirlist = os.listdir(os.getcwd())

  images = {'soilcontam':[], 'watercontam':[], 'poor':[], 'perfect':[], 'noclamp':[]}

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
        print 'brrreaking gash'
        break

  json.dump(images, open('images.txt','w'))
  
