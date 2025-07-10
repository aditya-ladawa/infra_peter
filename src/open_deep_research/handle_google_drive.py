import os
import io
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
from google.auth.transport.requests import Request # Import Request for creds.refresh()

# Google Drive API Scope
SCOPES = ['https://www.googleapis.com/auth/drive']

GOOGLE_CLIENT_SECRET_PATH = os.path.join('google_client_secret.com.json')

# Step 1: Authenticate and create a service object
def authenticate_google_drive():
    creds = None
    # The file token.json stores the user's access and refresh tokens
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                GOOGLE_CLIENT_SECRET_PATH, SCOPES)
            creds = flow.run_local_server(port=0)
        
        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    
    # Build the Drive service
    service = build('drive', 'v3', credentials=creds)
    return service

# Step 2: List files/folders in a specific folder (modified to find by name)
def find_folder_id_by_name(service, folder_name, parent_folder_id=None):
    """
    Finds the ID of a folder by its name within an optional parent folder.
    Returns the folder ID if found, otherwise None.
    """
    query_parts = [
        f"mimeType = 'application/vnd.google-apps.folder'",
        f"name = '{folder_name}'"
    ]
    if parent_folder_id:
        query_parts.append(f"'{parent_folder_id}' in parents")
    else:
        # If no parent_folder_id is given, search in 'My Drive' (root)
        query_parts.append(f"'root' in parents")
    
    query = " and ".join(query_parts)
    
    results = service.files().list(q=query, fields="files(id, name)").execute()
    items = results.get('files', [])

    if items:
        # Assuming unique folder names within a parent, return the first one
        print(f"Found folder '{folder_name}' with ID: {items[0]['id']}")
        return items[0]['id']
    else:
        print(f"Folder '{folder_name}' not found.")
        return None




def list_files(service, folder_id):
    query = f"'{folder_id}' in parents"
    results = service.files().list(q=query, fields="nextPageToken, files(id, name)").execute()
    items = results.get('files', [])
    
    # Always return a list
    return items

# Step 3: Create a folder in Google Drive
def create_folder(service, folder_name, parent_folder_id=None):
    file_metadata = {
        'name': folder_name,
        'mimeType': 'application/vnd.google-apps.folder'
    }
    if parent_folder_id:
        file_metadata['parents'] = [parent_folder_id]

    # Check if folder already exists in the parent to avoid duplicates
    existing_folder_id = find_folder_id_by_name(service, folder_name, parent_folder_id)
    if existing_folder_id:
        print(f'Folder "{folder_name}" already exists with ID: {existing_folder_id}. Reusing it.')
        return existing_folder_id

    folder = service.files().create(body=file_metadata, fields='id').execute()
    print(f'Folder created with ID: {folder["id"]}')
    return folder['id']

# Step 4: Upload a file to a specific folder
def upload_file(service, file_path, folder_id):
    file_metadata = {
        'name': os.path.basename(file_path),
        'parents': [folder_id]
    }
    media = MediaFileUpload(file_path, resumable=True)
    
    file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    print(f'File uploaded with ID: {file["id"]}')
    return file['id']

# Step 5: Download a file from a specific folder
def download_file(service, file_id, destination_path):
    request = service.files().get_media(fileId=file_id)
    fh = io.FileIO(destination_path, 'wb')
    downloader = MediaIoBaseDownload(fh, request)

    done = False
    while done is False:
        status, done = downloader.next_chunk()
        print(f"Download {int(status.progress() * 100)}%.")
    print(f"Download complete: {destination_path}")

# Step 6: Delete a file or folder
def delete_file(service, file_id):
    service.files().delete(fileId=file_id).execute()
    print(f'File or folder with ID: {file_id} deleted.')

# Step 7: Update file metadata (e.g., rename a file)
def update_file(service, file_id, new_name):
    file_metadata = {'name': new_name}
    updated_file = service.files().update(fileId=file_id, body=file_metadata).execute()
    print(f'File updated: {updated_file["name"]}')
    return updated_file



if __name__ == '__main__':
    service = authenticate_google_drive()

#     # My Drive is the root folder
#     my_drive_id = 'root'

#     # 1. Find the 'channel_automations' folder ID within 'My Drive'
#     channel_automations_folder_id = find_folder_id_by_name(service, 'channel_automations', my_drive_id)
    
#     if not channel_automations_folder_id:
#         print("Error: 'channel_automations' folder not found in My Drive. Please ensure it exists.")
#     else:
#         # 2. Find the 'peterAI' folder ID within 'channel_automations'
#         peterai_folder_id = find_folder_id_by_name(service, 'peterAI', channel_automations_folder_id)

#         if not peterai_folder_id:
#             print("Error: 'peterAI' folder not found inside 'channel_automations'. Please ensure it exists.")
#         else:
#             # 3. Create 'SCRIPTS' inside 'peterAI'
#             scripts_folder_id = create_folder(service, 'SCRIPTS', peterai_folder_id)
#             print(f'New SCRIPTS Folder ID: {scripts_folder_id}')
            
#             # Example: List files in the newly created SCRIPTS folder
#             folder_files = list_files(service, scripts_folder_id)

#             # Example: Upload a file to the folder
#             file_id = upload_file(service, 'path/to/local/file.txt', scripts_folder_id)
            
#             # # Example: Download the uploaded file
#             # # download_file(service, file_id, 'path/to/save/file.txt')
            
#             # # Example: Update the file name
#             # # updated_file = update_file(service, file_id, 'updated_file.txt')

#             # # Example: Delete a file or folder
#             # # delete_file(service, file_id)






















