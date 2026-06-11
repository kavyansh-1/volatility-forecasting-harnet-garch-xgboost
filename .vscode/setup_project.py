import os
FOLDERS = [
    "data/raw", ## for original downloads 
    "data/processed", ## cleaned version and the version which are engineered
    "notebooks", ##Jupyter notebooks on which we will work
    "src", #Reusable Python modules which we will import later
    "models",  ##Saved model files   
    "plots", ## All the generated graphs and visualizations will go here
    "reports", # Summary tables and the performance CSVs


]

for folder in FOLDERS:
    os.makedirs(folder, exist_ok=True) #exist_ok= True means no crash if yhe folder pre exits
    print(f"✓  {folder}/ ")
print("\nProject Structure ready. ")