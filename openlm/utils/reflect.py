import os
import importlib


def register_modules_in_package(package, file):
    # Get the path to the package folder
    package_folder = os.path.dirname(file)

    # Loop through each file in the package folder and import it if it's a Python module
    for file in os.listdir(package_folder):
        file: str
        if file.endswith(".py") and not file.startswith("__"):
            module_name = file[:-3]  # Remove the .py extension
            module_path = f"{package}.{module_name}"  # Get the full path to the module
            importlib.import_module(module_path)
