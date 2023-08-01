# Download image data from box
# Imports
from boxsdk import OAuth2, Client
import os
import numpy as np

access_token='' # changes

# set up the authication 
def box_auth_setup(access_token, start=False):
    if start:
        oauth =OAuth2(
            client_id= '',
            client_secret = '',

            access_token=access_token
            )

        client = Client(oauth)
        user = client.user().get()
        print('Current User: ', user.id)

        return client

def get_box_items(folderID,client):
    folder = client.folder(folder_id= folderID).get()
    print(f'Folder "{folder.name}" has {folder.item_collection["total_count"]} items in it')
    items = client.folder(folder_id=folderID).get_items()
    return items

def dl_box_certain_items(client, items,file_id, saveto):
    for idx, item in enumerate(items):
          if file_id in item.name.lower():
              print('Idx', idx)

              item_content = client.file(item.id).get()
              with open(os.path.join(str(saveto)+item.name), 'wb') as open_file:
                item_content.download_to(open_file)
                open_file.close()


# download all files



def dl_box_data(items, client, saveto): #
  files = []
  for idx, item in enumerate(items):
      #print(f'{item.type.capitalize()} {item.id} is named "{item.name}"', idx)
      item_content = client.file(item.id).get()

      print('Idx', idx)

      with open(os.path.join(str(saveto),item.name), 'wb') as open_file:
        item_content.download_to(open_file)
        open_file.close()

      # label and file name added to list dict
      if item.name[0:4] == 'IDSW':
        if 'test' not in item.name.lower():
            if item.name[6] == '1':
                files.append({'label': 0, 'file_name' : item.name})
            if item.name[6] == '2':
                files.append({'label': 1, 'file_name' : item.name})
            if item.name[6] =='3':
                files.append({'label': 2, 'file_name' : item.name})
            if item.name[6] =='4':
                files.append({'label': 3, 'file_name' : item.name})
            if item.name[6] =='5':
                files.append({'label': 4, 'file_name' : item.name})
            if item.name[6] =='6':
                files.append({'label': 5, 'file_name' : item.name})
            if item.name[6] =='7':
                files.append({'label': 6, 'file_name' : item.name})
            if item.name[6] =='8':
                files.append({'label': 7, 'file_name' : item.name})
            if item.name[6] =='9':
                files.append({'label': 8, 'file_name' : item.name})
            if item.name[6] =='10':
                files.append({'label': 9, 'file_name' : item.name})
            if item.name[6] =='11':
                files.append({'label': 10, 'file_name' : item.name})
  else:
        # check for any missed item
        print(item.name, 'xxxxxxxxxxxxxx')

  return files


def set_label_image_lists(file_path):
    labels= []
    images = []
    for file in os.listdir(file_path):
        if file[0:4] == 'IDSW':
            i=int(file[5:7]) -1
            i = str(i)
            labels.append(i)

    for i in os.listdir(file_path):
        if i[0:4] == 'IDSW':
            j=file_path+i
            images.append(j)

    label_arr =np.array(labels)
    image_arr = np.array(images)
    return label_arr, image_arr