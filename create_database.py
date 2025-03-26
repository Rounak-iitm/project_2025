pip install -r requirements.txt
import os
import pandas as pd

def create_database_csv(image_folder, output_csv):
    data = []
    for branch_folder in os.listdir(image_folder):
        branch_path = os.path.join(image_folder, branch_folder)
        if not os.path.isdir(branch_path):
            continue
        
        branch = branch_folder.replace('_', ' ')
        
        for image_file in os.listdir(branch_path):
            if not image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            
            name = os.path.splitext(image_file)[0].replace('_', ' ')
            image_path = os.path.abspath(os.path.join(branch_path, image_file))
            
            data.append({
                'name': name,
                'branch': branch,
                'image_path': image_path
            })
    
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"Created database with {len(df)} entries, saved to {output_csv}")

if __name__ == "__main__":
    create_database_csv('path_to_image_folder', 'face_database.csv')
