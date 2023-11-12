import os
import subprocess

# Define the path to your package
package_path = './chatstream'

# Create a virtual environment
subprocess.run(['python3', '-m', 'venv', 'test_env'])

# Activate the virtual environment
activate = '. test_env/bin/activate'

# Install your package
subprocess.run([activate, '&&', 'pip', 'install', '-e', package_path], shell=True)

# Run a test script
# Replace 'test_script.py' with the path to your test script
subprocess.run([activate, '&&', 'python', './examples/basic/app.py'], shell=True)

# Deactivate the virtual environment
subprocess.run(['deactivate'], shell=True)
