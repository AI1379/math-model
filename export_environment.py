import yaml
import subprocess

result = subprocess.run(['conda', 'env', 'export'], capture_output=True, text=True)
data = yaml.safe_load(result.stdout)

if 'prefix' in data:
    del data['prefix']
    
if 'name' in data:
    data['name'] = '.conda'

with open('environment.yml', 'w') as f:
    yaml.dump(data, f, default_flow_style=False)
