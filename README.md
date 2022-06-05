# Remove text from bulk images
Remove text from multiple images at once.
This script allows the user to remove texts from multiple images automatically
or without human assistance using keras ocr and opencv inpainting

It accepts 3 optional arguments - images directory, output directory and batch size.
Default image directory: Folder named 'images' in the same directory as the script
Default output directory: Output folder inside same directory
Default batch size: 1

This script requires the libraires in the requirements.txt to be installed within the Python
environment you are running this script in.

 
 # How to use:
 
 Install the requirements in the 'requirements.txt' within the Python environment you are running this project in.
 Run "text_remove.py" if you have the images folder and script in the same directory.
 Or run it with a command like this,
 python text_remove.py -i 'path/to/images' -o 'path/to/output/' -b 2
 
 The arguments are optional which means it will run without any or no arguments.
 
