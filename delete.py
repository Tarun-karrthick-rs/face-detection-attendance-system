import os
import pickle
import argparse

data_directory = 'data/'

def delete_specific_data(name_to_delete):

    names_path = os.path.join(data_directory, 'names.pkl')
    faces_path = os.path.join(data_directory, 'face_data.pkl')

    if not os.path.exists(names_path):
        print("'names.pkl' not found in the directory.")
        return
    
    with open(names_path, 'rb') as f:
        names = pickle.load(f)
    if name_to_delete not in names:
        print(f"'{name_to_delete}' not found in the 'names.pkl' dataset.")
        return
    
    indices_to_remove = [i for i, name in enumerate(names) if name == name_to_delete]
    
    with open(faces_path, 'rb') as f:
        faces = pickle.load(f)
    
    faces = [face for i, face in enumerate(faces) if i not in indices_to_remove]
    
    names = [name for name in names if name != name_to_delete]

    with open(names_path, 'wb') as f:
        pickle.dump(names, f)
    
    with open(faces_path, 'wb') as f:
        pickle.dump(faces, f)

    print(f"'{name_to_delete}' and corresponding face data have been deleted successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Delete a specific name and its corresponding face data.")
    parser.add_argument("--delete", type=str, help="Enter the name to delete (it should match the name in 'names.pkl').", required=True)

    args = parser.parse_args()
    delete_specific_data(args.delete)
