
import os
import yaml
import re

pattern = re.compile(r"\$\{(\w+)(?::-(.*?))?\}")

def load_config_mock(content):
    def replace_env_var(match):
        var_name = match.group(1)
        default = match.group(2)
        val = os.environ.get(var_name)
        if val is not None:
            return val
        return default if default is not None else match.group(0)
    
    expanded_content = pattern.sub(replace_env_var, content)
    return yaml.safe_load(expanded_content)

yaml_content = """
email_recipients:
  - "${RECIPIENT_EMAIL:-your-email@example.com}"
"""

os.environ["RECIPIENT_EMAIL"] = "a@b.com,c@d.com"
config = load_config_mock(yaml_content)
print(f"Recipients: {config['email_recipients']}")
