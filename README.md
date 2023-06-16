# Vehicle-Routing-Problem
This repo contains a source code in Python as well C/C++ and CUDA for VRP

To compile the program both python and CUDA you will need following tools and libraries.

1. Python 3 and above with numpy, networkx and matplotlib
2. CUDA 8.0 Toolkit, NVidia GPU, Visual Studio 2015(Any edition)
3. Microsoft Windows 7 and above OS

Compiling and Executing Python Program:

1. Keep the vrp.py file and datasets in same folder
2. Run command in command prompt on Windows "python vrp.py dataset_name.vrp"

Compiling and Executing CUDA Program:

1. Create new NVidia project in Visual Studio and update project properties with all necessary included headers
2. Add vrp.cu as source in project
3. Set thread define values to MAX supported by your NVidia GPU
4. Build the program
5. Go to your project folder and keep all datasets and routesGraphGen.py in Debug folder
5. Run in Bash with the following command "YourProjectName.exe dataset_name.vrp"
6. This will generate routes.txt, an output file
7. To generate graph run the python script "python routesGraphGen.py" in command prompt

Please contact Prasad Pandit (prasadp4009@gmail.com) if you have any questions.

Thank You..!!!!!!!
